defmodule Lux.LLM.RouterTest do
  use UnitCase, async: true

  @moduletag :capture_log

  alias Lux.LLM.Cache
  alias Lux.LLM.Metrics
  alias Lux.LLM.Provider
  alias Lux.LLM.Registry
  alias Lux.LLM.Response
  alias Lux.LLM.Router
  alias Lux.LLM.Selector

  defmodule OkProvider do
    @moduledoc false
    @behaviour Lux.LLM

    @impl true
    def call(prompt, _tools, options) do
      if pid = options[:test_pid], do: send(pid, {:called, __MODULE__, options[:model]})
      {:ok, %Lux.LLM.Response{content: "ok: " <> prompt, finish_reason: "stop"}}
    end
  end

  defmodule ErrorProvider do
    @moduledoc false
    @behaviour Lux.LLM

    @impl true
    def call(_prompt, _tools, options) do
      if pid = options[:test_pid], do: send(pid, {:called, __MODULE__, options[:model]})
      {:error, "rate limited"}
    end
  end

  defmodule RaisingProvider do
    @moduledoc false
    @behaviour Lux.LLM

    @impl true
    def call(_prompt, _tools, _options), do: raise("boom")
  end

  setup context do
    metrics = Module.concat(__MODULE__, "Metrics#{context.line}")
    cache = Module.concat(__MODULE__, "Cache#{context.line}")
    registry = Module.concat(__MODULE__, "Registry#{context.line}")

    start_supervised!({Metrics, name: metrics})
    start_supervised!({Cache, name: cache, max_size: 2})
    start_supervised!({Registry, name: registry, providers: []})

    %{metrics: metrics, cache: cache, registry: registry}
  end

  defp provider(name, module, attrs \\ []) do
    Provider.new([name: name, module: module, options: %{test_pid: self()}] ++ attrs)
  end

  defp names(candidates), do: Enum.map(candidates, fn {p, m} -> {p.name, m.name} end)

  describe "Provider" do
    test "normalizes models and options" do
      provider =
        Provider.new(
          name: :p,
          module: OkProvider,
          options: [temperature: 0.1],
          models: [[name: "m", input_cost: 1.0, output_cost: 2.0, capabilities: [:tools]]]
        )

      assert %Provider{options: %{temperature: 0.1}, models: [%Provider.Model{name: "m"}]} =
               provider

      assert Provider.supports?(hd(provider.models), [:tools])
      refute Provider.supports?(hd(provider.models), [:vision])
      assert Provider.estimate_cost(hd(provider.models), 1000, 500) == 2.0
    end

    test "estimates tokens" do
      assert Provider.estimate_tokens("abcdefgh") == 2
      assert Provider.estimate_tokens(%Response{content: "abcd"}) == 1
      assert Provider.estimate_tokens(nil) == 0
    end
  end

  describe "Registry" do
    test "registers, lists by priority, enables and disables", %{registry: registry} do
      {:ok, _} = Registry.register(registry, name: :low, module: OkProvider, priority: 1)
      {:ok, _} = Registry.register(registry, name: :high, module: OkProvider, priority: 10)

      assert [:high, :low] == registry |> Registry.list() |> Enum.map(& &1.name)
      assert :ok == Registry.disable(registry, :high)
      assert {:ok, %Provider{enabled: false}} = Registry.get(registry, :high)
      assert :ok == Registry.enable(registry, :high)
      assert {:error, :not_found} == Registry.disable(registry, :missing)

      assert :ok == Registry.unregister(registry, :low)
      assert {:error, :not_found} == Registry.get(registry, :low)
    end
  end

  describe "Selector" do
    setup do
      cheap = %{name: "cheap", input_cost: 0.1, output_cost: 0.1, quality: 1}
      smart = %{
        name: "smart",
        input_cost: 1.0,
        output_cost: 1.0,
        quality: 9,
        capabilities: [:tools]
      }

      providers = [
        Provider.new(name: :a, module: OkProvider, priority: 1, models: [smart]),
        Provider.new(name: :b, module: OkProvider, priority: 5, models: [cheap]),
        Provider.new(name: :c, module: OkProvider, priority: 9, enabled: false, models: [cheap])
      ]

      %{providers: providers}
    end

    test "ranks by strategy and skips disabled providers", %{providers: providers} do
      assert [{:b, "cheap"}, {:a, "smart"}] == names(Selector.select(providers))
      assert [{:b, "cheap"}, {:a, "smart"}] == names(Selector.select(providers, strategy: :cost))
      assert [{:a, "smart"}, {:b, "cheap"}] ==
               names(Selector.select(providers, strategy: :quality))
    end

    test "filters by capabilities, cost, model and exclusions", %{providers: providers} do
      assert [{:a, "smart"}] == names(Selector.select(providers, capabilities: [:tools]))
      assert [{:b, "cheap"}] == names(Selector.select(providers, max_cost: 0.5))
      assert [{:a, "smart"}] == names(Selector.select(providers, model: "smart"))
      assert [{:a, "smart"}] == names(Selector.select(providers, exclude: [:b]))
    end

    test "uses metrics for latency ranking and failing models", %{providers: providers} do
      metrics = %{
        {:b, "cheap"} => %{calls: 4, avg_latency_ms: 500.0, consecutive_failures: 0},
        {:a, "smart"} => %{calls: 4, avg_latency_ms: 50.0, consecutive_failures: 0}
      }

      assert [{:a, "smart"}, {:b, "cheap"}] ==
               names(Selector.select(providers, %{strategy: :latency}, metrics))

      failing = put_in(metrics, [{:b, "cheap"}, :consecutive_failures], 3)
      assert [{:a, "smart"}, {:b, "cheap"}] == names(Selector.select(providers, %{}, failing))
    end
  end

  describe "route/3" do
    test "uses the highest ranked provider and passes the model", ctx do
      providers = [
        provider(:primary, OkProvider, priority: 10, models: [%{name: "m1"}]),
        provider(:secondary, ErrorProvider, priority: 1)
      ]

      assert {:ok, %Response{content: "ok: hi"}, meta} =
               Router.route("hi", [], providers: providers, metrics: ctx.metrics)

      assert %{provider: :primary, model: "m1", cached: false, attempts: [%{status: :ok}]} = meta
      assert_received {:called, OkProvider, "m1"}
      refute_received {:called, ErrorProvider, _}
    end

    test "falls back to the next provider on errors and exceptions", ctx do
      providers = [
        provider(:raising, RaisingProvider, priority: 10),
        provider(:failing, ErrorProvider, priority: 5),
        provider(:working, OkProvider, priority: 1)
      ]

      assert {:ok, _response, meta} =
               Router.route("hi", [], providers: providers, metrics: ctx.metrics)

      assert meta.provider == :working

      assert [
               %{provider: :raising, status: :error, error: {:exception, "boom"}},
               %{provider: :failing, status: :error, error: "rate limited"},
               %{provider: :working, status: :ok}
             ] = meta.attempts
    end

    test "does not fall back when disabled", ctx do
      providers = [
        provider(:failing, ErrorProvider, priority: 5),
        provider(:working, OkProvider, priority: 1)
      ]

      assert {:error, {:all_providers_failed, [%{provider: :failing}]}} =
               Router.route("hi", [], providers: providers, fallback: false, metrics: ctx.metrics)

      refute_received {:called, OkProvider, _}
    end

    test "returns errors when nothing is available or everything fails", ctx do
      assert {:error, :no_providers_available} =
               Router.route("hi", [], registry: ctx.registry, metrics: ctx.metrics)

      providers = [provider(:failing, ErrorProvider)]

      assert {:error, "All LLM providers failed: :failing/nil: \"rate limited\""} =
               Router.call("hi", [], providers: providers, metrics: ctx.metrics)
    end

    test "reads providers from the registry", ctx do
      Registry.register(ctx.registry, provider(:registered, OkProvider))
      assert {:ok, %Response{}} = Router.call("hi", [], registry: ctx.registry, metrics: false)
      assert_received {:called, OkProvider, nil}
    end

    test "records cost and performance metrics", ctx do
      model = %{name: "m", input_cost: 1.0, output_cost: 2.0}

      providers = [
        provider(:failing, ErrorProvider, priority: 5, models: [model]),
        provider(:working, OkProvider, priority: 1, models: [model])
      ]

      {:ok, _, _} = Router.route("12345678", [], providers: providers, metrics: ctx.metrics)

      assert %{calls: 1, failures: 1, consecutive_failures: 1, total_cost: +0.0} =
               Metrics.get(ctx.metrics, {:failing, "m"})

      # "12345678" is 2 input tokens, "ok: 12345678" is 3 output tokens.
      assert %{calls: 1, successes: 1, input_tokens: 2, output_tokens: 3, total_cost: cost} =
               Metrics.get(ctx.metrics, {:working, "m"})

      assert_in_delta cost, 2 / 1000 * 1.0 + 3 / 1000 * 2.0, 1.0e-9

      summary = Metrics.summary(ctx.metrics)
      assert %{calls: 2, successes: 1, failures: 1} = summary
      assert %{working: %{calls: 1}, failing: %{calls: 1}} = summary.providers

      Metrics.reset(ctx.metrics)
      assert %{calls: 0} = Metrics.summary(ctx.metrics)
    end

    test "caches successful responses when enabled", ctx do
      providers = [provider(:working, OkProvider)]
      opts = [providers: providers, metrics: false, cache: true, cache_server: ctx.cache]

      assert {:ok, response, %{cached: false}} = Router.route("hi", [], opts)
      assert_received {:called, OkProvider, _}

      assert {:ok, ^response, %{cached: true, provider: :working}} = Router.route("hi", [], opts)
      refute_received {:called, OkProvider, _}

      assert {:ok, _, %{cached: false}} = Router.route("other", [], opts)
    end
  end

  describe "Cache" do
    test "expires entries and evicts the oldest when full", %{cache: cache} do
      Cache.put(cache, :a, 1, 0)
      assert :miss == Cache.get(cache, :a)

      Cache.put(cache, :a, 1)
      Cache.put(cache, :b, 2)
      Cache.put(cache, :c, 3)

      assert :miss == Cache.get(cache, :a)
      assert {:ok, 2} == Cache.get(cache, :b)
      assert {:ok, 3} == Cache.get(cache, :c)
      assert Cache.size(cache) == 2

      Cache.clear(cache)
      assert Cache.size(cache) == 0
    end
  end
end
