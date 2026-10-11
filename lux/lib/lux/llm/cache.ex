defmodule Lux.LLM.Cache do
  @moduledoc """
  In-memory response cache with TTL and a bounded size, used by `Lux.LLM.Router`
  when called with `cache: true`.

  A default cache named `Lux.LLM.Cache` is started with the Lux application.
  """

  use Agent

  @default_ttl :timer.minutes(5)
  @default_max_size 1_000

  @doc """
  Starts a cache.

  ## Options

    * `:name` - registered name (defaults to `#{inspect(__MODULE__)}`)
    * `:max_size` - maximum number of entries; the oldest entry is evicted when full
  """
  def start_link(opts \\ []) do
    max_size = Keyword.get(opts, :max_size, @default_max_size)
    name = Keyword.get(opts, :name, __MODULE__)
    Agent.start_link(fn -> %{entries: %{}, max_size: max_size} end, name: name)
  end

  @doc "Builds a cache key from a prompt, tools and options."
  @spec key(term(), list(), map()) :: binary()
  def key(prompt, tools, options) do
    tool_names = Enum.map(tools, &tool_name/1)
    :crypto.hash(:sha256, :erlang.term_to_binary({prompt, tool_names, options}))
  end

  @doc "Fetches a non-expired entry."
  @spec get(Agent.agent(), term()) :: {:ok, term()} | :miss
  def get(cache \\ __MODULE__, key) do
    now = now()

    Agent.get_and_update(cache, fn state ->
      case Map.fetch(state.entries, key) do
        {:ok, {value, expires_at, _inserted_at}} when expires_at > now ->
          {{:ok, value}, state}

        {:ok, _expired} ->
          {:miss, %{state | entries: Map.delete(state.entries, key)}}

        :error ->
          {:miss, state}
      end
    end)
  end

  @doc "Stores a value for `ttl` milliseconds (defaults to five minutes)."
  @spec put(Agent.agent(), term(), term(), non_neg_integer()) :: :ok
  def put(cache \\ __MODULE__, key, value, ttl \\ @default_ttl) do
    now = now()

    Agent.update(cache, fn state ->
      entries =
        if map_size(state.entries) >= state.max_size and not Map.has_key?(state.entries, key) do
          evict_oldest(state.entries, now)
        else
          state.entries
        end

      inserted_at = System.unique_integer([:monotonic])
      %{state | entries: Map.put(entries, key, {value, now + ttl, inserted_at})}
    end)
  end

  @doc "Removes all entries."
  @spec clear(Agent.agent()) :: :ok
  def clear(cache \\ __MODULE__), do: Agent.update(cache, &%{&1 | entries: %{}})

  @doc "Returns the number of stored entries (including not-yet-evicted expired ones)."
  @spec size(Agent.agent()) :: non_neg_integer()
  def size(cache \\ __MODULE__), do: Agent.get(cache, &map_size(&1.entries))

  defp evict_oldest(entries, now) do
    live = Map.filter(entries, fn {_key, {_value, expires_at, _}} -> expires_at > now end)

    if map_size(live) < map_size(entries) do
      live
    else
      {oldest, _} = Enum.min_by(live, fn {_key, {_value, _expires_at, inserted}} -> inserted end)
      Map.delete(live, oldest)
    end
  end

  defp tool_name(%{name: name}), do: name
  defp tool_name(tool) when is_atom(tool), do: tool
  defp tool_name(tool), do: inspect(tool)

  defp now, do: System.monotonic_time(:millisecond)
end
