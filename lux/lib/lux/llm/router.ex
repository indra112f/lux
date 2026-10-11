defmodule Lux.LLM.Router do
  @moduledoc """
  Provider-agnostic LLM client with automatic model selection, fallback handling,
  cost/performance tracking and response caching.

  `Lux.LLM.Router` implements the `Lux.LLM` behaviour itself, so it can be used
  anywhere a single provider module is accepted (including as
  `config :lux, Lux.LLM, default_module: Lux.LLM.Router`).

  ## Usage

      Lux.LLM.Registry.register(
        name: :openai,
        module: Lux.LLM.OpenAI,
        priority: 10,
        models: [%{name: "gpt-4o-mini", input_cost: 0.00015, output_cost: 0.0006}]
      )

      Lux.LLM.Registry.register(
        name: :anthropic,
        module: Lux.LLM.Anthropic,
        priority: 5,
        models: [%{name: "claude-3-5-haiku-latest", input_cost: 0.0008, output_cost: 0.004}]
      )

      # Cheapest capable model first; falls back to the next candidate on errors.
      {:ok, response} = Lux.LLM.Router.call("Hello", [], %{strategy: :cost, cache: true})

      # Same, but also returns which provider/model answered and every attempt.
      {:ok, response, meta} = Lux.LLM.Router.route("Hello", [], strategy: :cost)

      Lux.LLM.Metrics.summary()

  ## Router options

  All options below are consumed by the router. Any other option is passed through
  to the provider module, merged on top of the provider's default `options`. The
  selected model name is passed as `:model`.

    * `:strategy`, `:capabilities`, `:model`, `:max_cost`, `:exclude`,
      `:failure_threshold` - selection criteria, see `Lux.LLM.Selector`. When tools are
      given and `:capabilities` is not, `[:tools]` is required.
    * `:fallback` - try the next candidate when a call fails (default `true`)
    * `:max_attempts` - maximum number of candidates to try (default: all)
    * `:providers` - explicit list of providers to use instead of the registry
    * `:registry` - registry server (default `Lux.LLM.Registry`)
    * `:metrics` - metrics server, or `false` to disable tracking (default `Lux.LLM.Metrics`)
    * `:cache` - cache successful responses (default `false`)
    * `:cache_server` - cache server (default `Lux.LLM.Cache`)
    * `:cache_ttl` - cache TTL in milliseconds (default five minutes)
  """

  @behaviour Lux.LLM

  alias Lux.LLM.Cache
  alias Lux.LLM.Metrics
  alias Lux.LLM.Provider
  alias Lux.LLM.Registry
  alias Lux.LLM.Selector

  require Logger

  @selection_keys [:strategy, :capabilities, :model, :max_cost, :exclude, :failure_threshold]
  @control_keys [
    :fallback,
    :max_attempts,
    :providers,
    :registry,
    :metrics,
    :cache,
    :cache_server,
    :cache_ttl
  ]
  @default_cache_ttl :timer.minutes(5)

  @type attempt :: %{
          provider: atom() | String.t(),
          model: String.t() | nil,
          status: :ok | :error,
          latency_ms: non_neg_integer(),
          error: term()
        }

  @type meta :: %{
          provider: atom() | String.t(),
          model: String.t() | nil,
          cached: boolean(),
          attempts: [attempt()]
        }

  @impl true
  def call(prompt, tools, options) do
    case route(prompt, tools, options) do
      {:ok, response, _meta} -> {:ok, response}
      {:error, reason} -> {:error, format_error(reason)}
    end
  end

  @doc """
  Like `call/3`, but also returns metadata about the provider and model that answered,
  whether the response was cached and every attempt made.
  """
  @spec route(Lux.LLM.prompt(), Lux.LLM.tools(), Lux.LLM.options()) ::
          {:ok, term(), meta()}
          | {:error, :no_providers_available}
          | {:error, {:all_providers_failed, [attempt()]}}
  def route(prompt, tools, options) do
    options = Map.new(options)
    {criteria, rest} = Map.split(options, @selection_keys)
    {control, passthrough} = Map.split(rest, @control_keys)
    criteria = Map.put_new(criteria, :capabilities, if(tools == [], do: [], else: [:tools]))

    with_cache(control, {prompt, tools, Map.merge(criteria, passthrough)}, fn ->
      metrics = metrics_snapshot(control)

      control
      |> providers()
      |> Selector.select(criteria, metrics)
      |> limit_attempts(control)
      |> attempt(prompt, tools, passthrough, control, [])
    end)
  end

  defp with_cache(%{cache: true} = control, {prompt, tools, key_options}, fun) do
    server = Map.get(control, :cache_server, Cache)
    key = Cache.key(prompt, tools, key_options)

    case Cache.get(server, key) do
      {:ok, {response, meta}} ->
        {:ok, response, %{meta | cached: true, attempts: []}}

      :miss ->
        with {:ok, response, meta} <- fun.() do
          ttl = Map.get(control, :cache_ttl, @default_cache_ttl)
          Cache.put(server, key, {response, meta}, ttl)
          {:ok, response, meta}
        end
    end
  end

  defp with_cache(_control, _key, fun), do: fun.()

  defp providers(%{providers: providers}), do: Enum.map(providers, &Provider.new/1)
  defp providers(control), do: Registry.list(Map.get(control, :registry, Registry))

  defp limit_attempts(candidates, control) do
    case control do
      %{fallback: false} -> Enum.take(candidates, 1)
      %{max_attempts: max} when is_integer(max) -> Enum.take(candidates, max)
      _ -> candidates
    end
  end

  defp attempt([], _prompt, _tools, _options, _control, []), do: {:error, :no_providers_available}

  defp attempt([], _prompt, _tools, _options, _control, attempts),
    do: {:error, {:all_providers_failed, Enum.reverse(attempts)}}

  defp attempt([{provider, model} | rest], prompt, tools, options, control, attempts) do
    provider_options =
      provider.options
      |> Map.merge(options)
      |> maybe_put_model(model.name)

    started = System.monotonic_time(:millisecond)
    result = safe_call(provider.module, prompt, tools, provider_options)
    latency_ms = System.monotonic_time(:millisecond) - started

    case result do
      {:ok, response} ->
        attempt = %{
          provider: provider.name,
          model: model.name,
          status: :ok,
          latency_ms: latency_ms,
          error: nil
        }

        track(control, model, attempt, prompt, response)

        meta = %{
          provider: provider.name,
          model: model.name,
          cached: false,
          attempts: Enum.reverse([attempt | attempts])
        }

        {:ok, response, meta}

      {:error, reason} ->
        attempt = %{
          provider: provider.name,
          model: model.name,
          status: :error,
          latency_ms: latency_ms,
          error: reason
        }

        track(control, model, attempt, nil, nil)

        if rest != [] do
          Logger.warning(
            "LLM provider #{inspect(provider.name)} (#{inspect(model.name)}) failed: " <>
              "#{inspect(reason)}; falling back"
          )
        end

        attempt(rest, prompt, tools, options, control, [attempt | attempts])
    end
  end

  defp maybe_put_model(options, nil), do: options
  defp maybe_put_model(options, name), do: Map.put(options, :model, name)

  defp safe_call(module, prompt, tools, options) do
    case module.call(prompt, tools, options) do
      {:ok, response} -> {:ok, response}
      {:error, reason} -> {:error, reason}
      other -> {:error, {:unexpected_response, other}}
    end
  rescue
    exception -> {:error, {:exception, Exception.message(exception)}}
  catch
    :exit, reason -> {:error, {:exit, reason}}
  end

  defp metrics_snapshot(control) do
    case metrics_server(control) do
      nil -> %{}
      server -> Metrics.all(server)
    end
  end

  # Failed calls are recorded without token usage or cost.
  defp track(control, model, attempt, prompt, response) do
    case metrics_server(control) do
      nil ->
        :ok

      server ->
        input_tokens = Provider.estimate_tokens(prompt)
        output_tokens = Provider.estimate_tokens(response)

        Metrics.record(
          server,
          Map.merge(attempt, %{
            input_tokens: input_tokens,
            output_tokens: output_tokens,
            cost: Provider.estimate_cost(model, input_tokens, output_tokens)
          })
        )
    end
  end

  defp metrics_server(control) do
    case Map.get(control, :metrics, Metrics) do
      false -> nil
      server -> if GenServer.whereis(server), do: server
    end
  end

  defp format_error(:no_providers_available), do: "No LLM providers available for this request"

  defp format_error({:all_providers_failed, attempts}) do
    details =
      Enum.map_join(attempts, "; ", fn attempt ->
        "#{inspect(attempt.provider)}/#{inspect(attempt.model)}: #{inspect(attempt.error)}"
      end)

    "All LLM providers failed: " <> details
  end
end
