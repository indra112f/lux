defmodule Lux.LLM.Selector do
  @moduledoc """
  Model selection logic for `Lux.LLM.Router`.

  Given a list of providers, selection criteria and (optionally) collected metrics,
  returns an ordered list of `{provider, model}` candidates. The router tries
  candidates in order, falling back to the next one when a call fails.

  ## Criteria

    * `:strategy` - how to rank candidates:
      * `:priority` (default) - provider priority, highest first
      * `:cost` - cheapest model first (input + output cost per 1,000 tokens)
      * `:quality` - highest quality score first
      * `:latency` - lowest observed average latency first (unseen models first,
        so they get measured)
    * `:capabilities` - capabilities every candidate model must support, e.g. `[:tools]`
    * `:model` - only consider models with this name
    * `:max_cost` - skip models whose combined per-1k cost exceeds this value
    * `:exclude` - provider names to skip
    * `:failure_threshold` - models with at least this many consecutive failures are
      moved to the end of the list (default `3`)

  Disabled providers are always skipped. Providers without declared models are
  treated as serving a single default model (`nil`, assumed to support `:tools` and
  `:json`), which lets the provider module pick its own configured default.
  """

  alias Lux.LLM.Provider
  alias Lux.LLM.Provider.Model

  @type candidate :: {Provider.t(), Model.t()}

  @default_failure_threshold 3

  @spec select([Provider.t()], map() | keyword(), map()) :: [candidate()]
  def select(providers, criteria \\ %{}, metrics \\ %{}) do
    criteria = Map.new(criteria)
    capabilities = Map.get(criteria, :capabilities, [])
    exclude = Map.get(criteria, :exclude, [])
    threshold = Map.get(criteria, :failure_threshold, @default_failure_threshold)

    {healthy, unhealthy} =
      providers
      |> Enum.filter(&(&1.enabled and &1.name not in exclude))
      |> Enum.flat_map(&candidates/1)
      |> Enum.filter(fn {_provider, model} ->
        Provider.supports?(model, capabilities) and model_allowed?(model, criteria)
      end)
      |> rank(Map.get(criteria, :strategy, :priority), metrics)
      |> Enum.split_with(fn candidate ->
        stats(metrics, candidate).consecutive_failures < threshold
      end)

    healthy ++ unhealthy
  end

  defp candidates(%Provider{models: []} = provider) do
    [{provider, %Model{name: nil, capabilities: [:tools, :json]}}]
  end

  defp candidates(%Provider{models: models} = provider), do: Enum.map(models, &{provider, &1})

  defp model_allowed?(model, criteria) do
    name_ok = criteria |> Map.get(:model) |> then(&(is_nil(&1) or &1 == model.name))

    cost_ok =
      case Map.get(criteria, :max_cost) do
        nil -> true
        max_cost -> combined_cost(model) <= max_cost
      end

    name_ok and cost_ok
  end

  # Enum.sort_by/3 is stable, so ties keep provider priority order from the first sort.
  defp rank(candidates, strategy, metrics) do
    by_priority = Enum.sort_by(candidates, fn {provider, _} -> provider.priority end, :desc)

    case strategy do
      :priority ->
        by_priority

      :cost ->
        Enum.sort_by(by_priority, fn {_, model} -> combined_cost(model) end, :asc)

      :quality ->
        Enum.sort_by(by_priority, fn {_, model} -> model.quality end, :desc)

      :latency ->
        Enum.sort_by(by_priority, &latency_rank(metrics, &1), :asc)

      other ->
        raise ArgumentError, "unknown selection strategy: #{inspect(other)}"
    end
  end

  defp latency_rank(metrics, candidate) do
    case stats(metrics, candidate) do
      %{calls: 0} -> 0
      %{avg_latency_ms: avg} -> avg
    end
  end

  defp stats(metrics, {provider, model}) do
    Map.get(metrics, {provider.name, model.name}, %{calls: 0, consecutive_failures: 0})
  end

  defp combined_cost(%Model{input_cost: input, output_cost: output}), do: input + output
end
