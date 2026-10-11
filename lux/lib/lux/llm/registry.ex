defmodule Lux.LLM.Registry do
  @moduledoc """
  Registry of `Lux.LLM.Provider`s used by `Lux.LLM.Router`.

  A default registry named `Lux.LLM.Registry` is started with the Lux application.
  Providers listed under `config :lux, Lux.LLM.Registry, providers: [...]` are
  registered at startup. Additional registries can be started with a custom `:name`.
  """

  use Agent

  alias Lux.LLM.Provider

  @doc """
  Starts a registry.

  ## Options

    * `:name` - registered name (defaults to `#{inspect(__MODULE__)}`)
    * `:providers` - providers to register initially (defaults to the application config)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)

    providers =
      Keyword.get_lazy(opts, :providers, fn ->
        :lux |> Application.get_env(__MODULE__, []) |> Keyword.get(:providers, [])
      end)

    initial =
      providers
      |> Enum.map(&Provider.new/1)
      |> Map.new(&{&1.name, &1})

    Agent.start_link(fn -> initial end, name: name)
  end

  @doc "Registers (or replaces) a provider."
  @spec register(Agent.agent(), Provider.t() | map() | keyword()) :: {:ok, Provider.t()}
  def register(registry \\ __MODULE__, attrs) do
    provider = Provider.new(attrs)
    Agent.update(registry, &Map.put(&1, provider.name, provider))
    {:ok, provider}
  end

  @doc "Removes a provider by name."
  @spec unregister(Agent.agent(), atom() | String.t()) :: :ok
  def unregister(registry \\ __MODULE__, name) do
    Agent.update(registry, &Map.delete(&1, name))
  end

  @doc "Fetches a provider by name."
  @spec get(Agent.agent(), atom() | String.t()) :: {:ok, Provider.t()} | {:error, :not_found}
  def get(registry \\ __MODULE__, name) do
    case Agent.get(registry, &Map.fetch(&1, name)) do
      {:ok, provider} -> {:ok, provider}
      :error -> {:error, :not_found}
    end
  end

  @doc "Lists all registered providers, highest priority first."
  @spec list(Agent.agent()) :: [Provider.t()]
  def list(registry \\ __MODULE__) do
    registry
    |> Agent.get(&Map.values/1)
    |> Enum.sort_by(& &1.priority, :desc)
  end

  @doc "Enables a provider so it is considered for routing."
  @spec enable(Agent.agent(), atom() | String.t()) :: :ok | {:error, :not_found}
  def enable(registry \\ __MODULE__, name), do: set_enabled(registry, name, true)

  @doc "Disables a provider so it is skipped during routing."
  @spec disable(Agent.agent(), atom() | String.t()) :: :ok | {:error, :not_found}
  def disable(registry \\ __MODULE__, name), do: set_enabled(registry, name, false)

  defp set_enabled(registry, name, enabled) do
    Agent.get_and_update(registry, fn providers ->
      case Map.fetch(providers, name) do
        {:ok, provider} -> {:ok, Map.put(providers, name, %{provider | enabled: enabled})}
        :error -> {{:error, :not_found}, providers}
      end
    end)
  end
end
