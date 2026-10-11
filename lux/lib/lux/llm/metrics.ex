defmodule Lux.LLM.Metrics do
  @moduledoc """
  Tracks cost and performance metrics for calls made through `Lux.LLM.Router`.

  Metrics are aggregated per `{provider, model}` pair and include call counts,
  failures, consecutive failures (used for health-aware routing), latency and
  estimated token usage and cost.

  A default metrics server named `Lux.LLM.Metrics` is started with the Lux application.
  """

  use Agent

  @type key :: {atom() | String.t(), String.t() | nil}

  @type stats :: %{
          calls: non_neg_integer(),
          successes: non_neg_integer(),
          failures: non_neg_integer(),
          consecutive_failures: non_neg_integer(),
          total_latency_ms: non_neg_integer(),
          avg_latency_ms: float(),
          input_tokens: non_neg_integer(),
          output_tokens: non_neg_integer(),
          total_cost: float(),
          last_error: term()
        }

  @empty %{
    calls: 0,
    successes: 0,
    failures: 0,
    consecutive_failures: 0,
    total_latency_ms: 0,
    avg_latency_ms: 0.0,
    input_tokens: 0,
    output_tokens: 0,
    total_cost: 0.0,
    last_error: nil
  }

  def start_link(opts \\ []) do
    Agent.start_link(fn -> %{} end, name: Keyword.get(opts, :name, __MODULE__))
  end

  @doc """
  Records the outcome of a single call.

  `entry` must contain `:provider`, `:model`, `:status` (`:ok` or `:error`) and
  `:latency_ms`, and may contain `:input_tokens`, `:output_tokens`, `:cost` and `:error`.
  """
  @spec record(Agent.agent(), map()) :: :ok
  def record(server \\ __MODULE__, entry) do
    key = {entry.provider, entry.model}

    Agent.update(server, fn state ->
      Map.update(state, key, apply_entry(@empty, entry), &apply_entry(&1, entry))
    end)
  end

  @doc "Returns the stats for every `{provider, model}` pair."
  @spec all(Agent.agent()) :: %{key() => stats()}
  def all(server \\ __MODULE__), do: Agent.get(server, & &1)

  @doc "Returns the stats for a `{provider, model}` pair (empty stats if unseen)."
  @spec get(Agent.agent(), key()) :: stats()
  def get(server \\ __MODULE__, key), do: Agent.get(server, &Map.get(&1, key, @empty))

  @doc """
  Returns aggregated totals across all providers and models, plus per-provider totals
  under `:providers`.
  """
  @spec summary(Agent.agent()) :: map()
  def summary(server \\ __MODULE__) do
    stats = all(server)

    providers =
      stats
      |> Enum.group_by(fn {{provider, _model}, _} -> provider end, fn {_, s} -> s end)
      |> Map.new(fn {provider, list} -> {provider, merge(list)} end)

    stats |> Map.values() |> merge() |> Map.put(:providers, providers)
  end

  @doc "Clears all recorded metrics."
  @spec reset(Agent.agent()) :: :ok
  def reset(server \\ __MODULE__), do: Agent.update(server, fn _ -> %{} end)

  defp apply_entry(stats, entry) do
    calls = stats.calls + 1
    total_latency = stats.total_latency_ms + entry.latency_ms

    stats =
      %{
        stats
        | calls: calls,
          total_latency_ms: total_latency,
          avg_latency_ms: total_latency / calls,
          input_tokens: stats.input_tokens + Map.get(entry, :input_tokens, 0),
          output_tokens: stats.output_tokens + Map.get(entry, :output_tokens, 0),
          total_cost: stats.total_cost + Map.get(entry, :cost, 0.0)
      }

    case entry.status do
      :ok ->
        %{stats | successes: stats.successes + 1, consecutive_failures: 0}

      :error ->
        %{
          stats
          | failures: stats.failures + 1,
            consecutive_failures: stats.consecutive_failures + 1,
            last_error: Map.get(entry, :error)
        }
    end
  end

  defp merge(list) do
    totals =
      Enum.reduce(list, Map.delete(@empty, :last_error), fn s, acc ->
        %{
          acc
          | calls: acc.calls + s.calls,
            successes: acc.successes + s.successes,
            failures: acc.failures + s.failures,
            consecutive_failures: max(acc.consecutive_failures, s.consecutive_failures),
            total_latency_ms: acc.total_latency_ms + s.total_latency_ms,
            input_tokens: acc.input_tokens + s.input_tokens,
            output_tokens: acc.output_tokens + s.output_tokens,
            total_cost: acc.total_cost + s.total_cost
        }
      end)

    avg = if totals.calls > 0, do: totals.total_latency_ms / totals.calls, else: 0.0
    %{totals | avg_latency_ms: avg}
  end
end
