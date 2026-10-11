defmodule Lux.LLM.Provider do
  @moduledoc """
  Describes an LLM provider for the provider abstraction layer.

  A provider wraps any module implementing the `Lux.LLM` behaviour (for example
  `Lux.LLM.OpenAI`, `Lux.LLM.Anthropic` or `Lux.LLM.TogetherAI`) together with the
  metadata needed to route requests to it: the models it serves, their costs and
  capabilities, a priority and default options.

  ## Example

      Lux.LLM.Provider.new(
        name: :openai,
        module: Lux.LLM.OpenAI,
        priority: 10,
        options: %{temperature: 0.2},
        models: [
          %{name: "gpt-4o-mini", input_cost: 0.00015, output_cost: 0.0006,
            quality: 6, capabilities: [:tools, :json]},
          %{name: "gpt-4o", input_cost: 0.0025, output_cost: 0.01,
            quality: 9, capabilities: [:tools, :json, :vision]}
        ]
      )

  Costs are expressed per 1,000 tokens.
  """

  defmodule Model do
    @moduledoc """
    A model served by a `Lux.LLM.Provider`.

    * `:name` - model identifier passed to the provider as the `:model` option
    * `:input_cost` / `:output_cost` - cost per 1,000 input/output tokens
    * `:quality` - relative quality score (higher is better), used by the `:quality` strategy
    * `:capabilities` - list of atoms such as `:tools`, `:json` or `:vision`
    * `:context_window` - maximum context size in tokens, or `nil` when unknown
    """

    @type t :: %__MODULE__{
            name: String.t(),
            input_cost: number(),
            output_cost: number(),
            quality: number(),
            capabilities: [atom()],
            context_window: pos_integer() | nil
          }

    @enforce_keys [:name]
    defstruct name: nil,
              input_cost: 0.0,
              output_cost: 0.0,
              quality: 0,
              capabilities: [],
              context_window: nil

    @doc "Builds a model from a struct, map or keyword list."
    @spec new(t() | map() | keyword()) :: t()
    def new(%__MODULE__{} = model), do: model
    def new(attrs) when is_list(attrs), do: attrs |> Map.new() |> new()
    def new(attrs) when is_map(attrs), do: struct!(__MODULE__, attrs)
  end

  @type t :: %__MODULE__{
          name: atom() | String.t(),
          module: module(),
          models: [Model.t()],
          priority: integer(),
          enabled: boolean(),
          options: map()
        }

  @enforce_keys [:name, :module]
  defstruct name: nil,
            module: nil,
            models: [],
            priority: 0,
            enabled: true,
            options: %{}

  @doc """
  Builds a provider from a struct, map or keyword list. Models are normalized
  into `Lux.LLM.Provider.Model` structs and options into a map.
  """
  @spec new(t() | map() | keyword()) :: t()
  def new(%__MODULE__{} = provider), do: provider
  def new(attrs) when is_list(attrs), do: attrs |> Map.new() |> new()

  def new(attrs) when is_map(attrs) do
    provider = struct!(__MODULE__, attrs)

    if is_nil(provider.module) or not is_atom(provider.module) do
      raise ArgumentError, "provider :module must be a module, got: #{inspect(provider.module)}"
    end

    %{
      provider
      | models: Enum.map(provider.models, &Model.new/1),
        options: Map.new(provider.options)
    }
  end

  @doc "Returns true when the model provides every capability in `capabilities`."
  @spec supports?(Model.t(), [atom()]) :: boolean()
  def supports?(%Model{capabilities: available}, capabilities) do
    Enum.all?(capabilities, &(&1 in available))
  end

  @doc """
  Estimates the cost of a call given input and output token counts.
  """
  @spec estimate_cost(Model.t(), non_neg_integer(), non_neg_integer()) :: float()
  def estimate_cost(%Model{} = model, input_tokens, output_tokens) do
    input_tokens / 1000 * model.input_cost + output_tokens / 1000 * model.output_cost
  end

  @doc """
  Roughly estimates the number of tokens in a term (about four bytes per token).

  Providers in Lux do not consistently report token usage, so this heuristic is used
  for cost tracking. Treat the resulting costs as estimates.
  """
  @spec estimate_tokens(term()) :: non_neg_integer()
  def estimate_tokens(nil), do: 0
  def estimate_tokens(text) when is_binary(text), do: div(byte_size(text) + 3, 4)
  def estimate_tokens(%Lux.LLM.Response{content: content}), do: estimate_tokens(content)
  def estimate_tokens(%{payload: payload}), do: estimate_tokens(payload)
  def estimate_tokens(%{content: content}) when is_binary(content), do: estimate_tokens(content)
  def estimate_tokens(term), do: term |> inspect(limit: :infinity) |> estimate_tokens()
end
