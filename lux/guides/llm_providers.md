# LLM Provider Abstraction Layer

Lux ships with several LLM integrations (`Lux.LLM.OpenAI`, `Lux.LLM.Anthropic`,
`Lux.LLM.TogetherAI`, ...). They all implement the `Lux.LLM` behaviour, which is the
universal provider interface:

```elixir
@callback call(prompt, tools, options) :: {:ok, response} | {:error, reason}
```

`Lux.LLM.Router` builds on that interface to manage several providers at once:

| Module | Responsibility |
| --- | --- |
| `Lux.LLM.Provider` | Describes a provider: module, models, costs, capabilities, priority |
| `Lux.LLM.Registry` | Registers, lists, enables and disables providers |
| `Lux.LLM.Selector` | Ranks `{provider, model}` candidates (priority, cost, quality, latency) |
| `Lux.LLM.Router` | Calls the best candidate, falls back on failure, caches, tracks metrics |
| `Lux.LLM.Metrics` | Per provider/model calls, failures, latency, estimated tokens and cost |
| `Lux.LLM.Cache` | TTL response cache with a bounded size |

`Lux.LLM.Registry`, `Lux.LLM.Metrics` and `Lux.LLM.Cache` are started by the Lux
application.

## Registering providers

At runtime:

```elixir
Lux.LLM.Registry.register(
  name: :openai,
  module: Lux.LLM.OpenAI,
  priority: 10,
  models: [
    %{name: "gpt-4o-mini", input_cost: 0.00015, output_cost: 0.0006, quality: 6,
      capabilities: [:tools, :json]},
    %{name: "gpt-4o", input_cost: 0.0025, output_cost: 0.01, quality: 9,
      capabilities: [:tools, :json, :vision]}
  ]
)

Lux.LLM.Registry.register(
  name: :together,
  module: Lux.LLM.TogetherAI,
  priority: 1,
  options: %{temperature: 0.2}
)
```

Or in configuration (registered at startup):

```elixir
config :lux, Lux.LLM.Registry,
  providers: [
    [name: :openai, module: Lux.LLM.OpenAI, priority: 10,
     models: [[name: "gpt-4o-mini", input_cost: 0.00015, output_cost: 0.0006]]]
  ]
```

Costs are per 1,000 tokens and are only used for ranking and cost estimates — check
your providers' current pricing. A provider without `models` is called without a
`:model` option, so it uses its own configured default model.

## Calling through the router

```elixir
# Highest priority provider first, falling back on errors.
{:ok, response} = Lux.LLM.Router.call("Summarize this...", [], %{})

# Cheapest model that supports tools, at most two attempts, cached for one minute.
{:ok, response, meta} =
  Lux.LLM.Router.route("Which token is trending?", [MyApp.Prisms.Trending],
    strategy: :cost,
    max_attempts: 2,
    cache: true,
    cache_ttl: :timer.minutes(1)
  )

meta.provider  #=> :openai
meta.model     #=> "gpt-4o-mini"
meta.attempts  #=> [%{provider: :openai, status: :ok, latency_ms: 812, ...}]
```

Selection options: `:strategy` (`:priority`, `:cost`, `:quality`, `:latency`),
`:capabilities`, `:model`, `:max_cost`, `:exclude` and `:failure_threshold`. Models with
`failure_threshold` (default 3) consecutive failures are moved to the end of the list
until they succeed again. Any other option is passed through to the provider module.

Because the router implements `Lux.LLM`, it can also be made the default module:

```elixir
config :lux, Lux.LLM, default_module: Lux.LLM.Router
```

## Monitoring

```elixir
Lux.LLM.Metrics.summary()
#=> %{calls: 12, successes: 11, failures: 1, avg_latency_ms: 640.5,
#     input_tokens: 2_310, output_tokens: 4_020, total_cost: 0.0031,
#     providers: %{openai: %{...}, together: %{...}}, ...}

Lux.LLM.Metrics.get({:openai, "gpt-4o-mini"})
Lux.LLM.Metrics.reset()
```

Token counts are estimated from text length (about four bytes per token) because the
providers do not consistently report usage, so costs are approximations.
