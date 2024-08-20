---
title: 🧱 Structure from Language
date: '2024-8-20'
tags: 
- llm
- structure
- outlines
published: true
description: Three levels of generating structured outputs from language models.
image: /public/images/llm-structured-outputs/banner.png
twitter_image: /public/images/llm-structured-outputs/banner.png
---

Large Language Models are text token generators.
We give them unicode strings, they turn them into tokens (a sequence of numbers) and then predict a probability distribution over the next token in the sequence.
Let's take a real example, consider the following string:

```json
The cat sat on the
```

Using this as the input to a language model generator, we get the following output distribution (showing the top 10 probabilities):

<Image alt="the cat sat on the" src="/public/images/llm-structured-outputs/0.png" width={1000} height={500} />

A common way to sample the next token from the distribution is called **greedy decoding**.
This is simply selecting the token with the highest probability.

There are no rules or structural guarantees to the content of the output string.
Any token is valid at any position in the sequence.
Empirically we see remarkable adherence to the semantic grammar of language, but none of these rules have been predefined.
Instead they have been learned from data.
This may be fine for building a chatbot app, where we have a UI to directly display the LLM output to a human and strings are arbitrarily valid.

But what if we're composing an application where the LLM generator is near the start of our system?
Maybe it's helping us to extract information from unstructured documents, or it could be doing a zero shot classification task.
Now we require a contract between components to be able to build robust systems.

A standard approach is to write this contract in software.
For example, we could define a schema that both parties agree upon, which represents the data structure.
This schema would outline the required fields, their data types, and any validation rules.

LLMs don't have the deterministic benefits of inherent structure and schema that we're used to from traditional software engineering.
In the following three sections, we'll do what we can to regain some of these controls and make LLM systems more useful.

Assume we're building an animal classification system.
The new system should output JSON with the animal species like so:

```json
{
  "animal": "lion"
}
```

## Level 1 - Prompting

People have come up with creative ways to improve the likelihood the LLM will adhere to the desired structure by instruction prompting in natural language.

```json
Classify the document and output JSON with a single 'animal' key.
The species name should be the value.
Please please please output valid JSON, my career depends on it!
```

This is better than no instructions at all and may work, say 70% of the time.

Developers can try their best to fix JSON errors post-generation on the fly with libraries like [json_repair](https://github.com/mangiucugna/json_repair).
This may work in testing and for simple cases, but it's fundamentally fragile and hard to ship features using this method in a production codebase.

## Level 2 - Function calling (tool use)

We can do better by being explicit about the fact we want to generate JSON. We can do this by using a feature called function calling.
This gives the model the option to intelligently choose to output a JSON object containing arguments to call one or many functions. 
In practice, we create a JSON Schema and give that to the model alongside the prompt.

We can use [Pydantic](https://docs.pydantic.dev/latest/) to create a data model:

```python
from pydantic import BaseModel

class Animal(BaseModel):
    animal: str

print(Animal.model_json_schema())
```

...to generate [JSON Schema](https://json-schema.org/)

```
{'properties': {'animal': {'title': 'Animal', 'type': 'string'}}, 'required': ['animal'], 'title': 'Animal', 'type': 'object'}
```

...which we can use to call the LLM using the instructor library (using function calling under the hood):

```
import instructor
import datetime
from anthropic import AnthropicBedrock

client = instructor.from_anthropic(
    AnthropicBedrock(aws_region='us-east-1'),
    mode=instructor.Mode.ANTHROPIC_JSON,
    max_tokens=1024
)

resp = client.chat.completions.create(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[
        {
            "role": "user",
            "content": f"Classify this {document} and output JSON with a single 'animal' key."
        },
    ],
    response_model=Animal,
)
```

When we do this with the popular AI model providers APIs, we don't get any guarantees that the model will actually generate JSON compliant with the JSON schema we asked for 100% of the time.
OpenAI and Anthropic etc can try to optimise for this by training their models to be better at this task, but some percentage of the time we will still get malformed outputs.

Libraries like [Instructor](https://python.useinstructor.com/), [LangChain](https://python.langchain.com/v0.2/api_reference/core/output_parsers/langchain_core.output_parsers.json.JsonOutputParser.html) and others have developer tools that help to bridge this gap.
Typically this is done by defining the output schema in Pydantic, serialising it to JSON Schema and then validating the LLM response fits the data model explicitly.
In case it doesn't comply, the API call can be re-tried with the added context of the actual validation error.
These libraries help a lot and save developers time in not reimplementing the data modelling, prompting, retries and validation logic.

## Level 3 - Structured Output

We can do even better if we pull some tricks in the way tokens are sampled from the language model.
This is a family of techniques called constrained decoding.
Remember that the fundamental challenge is that the definition of a valid token according to a JSON schema is a function of the position in the output sequence.
As an example, imagine we're part way through a generation:

```
{"ani
```

Our LLM predicts this distribution:

<Image alt="1" src="/public/images/llm-structured-outputs/1.png" width={1000} height={500} />

Notice that only a subset of possible tokens are valid JSON according to the JSON Schema above.
Invalid tokens are greyed out.

We decide to take the token of highest probability append it to the sequence,

```
{"animal
```

...and then generate again...

<Image alt="2" src="/public/images/llm-structured-outputs/2.png" width={1000} height={500} />

```
{"animal":
```

...and again...

<Image alt="3" src="/public/images/llm-structured-outputs/3.png" width={1000} height={500} />

```
{"animal":"
```

...and again...

<Image alt="4" src="/public/images/llm-structured-outputs/4.png" width={1000} height={500} />

```
{"animal":"dog
```

...and once more...

<Image alt="5" src="/public/images/llm-structured-outputs/5.png" width={1000} height={500} />

now until we have a valid JSON object according to the schema.

```json
{ "animal": "dog" }
```

Notice that at each step, we only allow the LLM to select valid tokens according to the sequence thus far and the JSON Schema.
The definition of **valid** changes based on the values of previous tokens.
All other tokens (shown greyed out) were masked when sampling from the distribution.
When creating these visuals, I manually identified the invalid tokens.
We need a programmatic and dynamic way to define what is a valid next token to implement constrained decoding.

One way this can be achieved is by converting our JSON Schema into a context-free grammar (CFG).
This is a formal way to specify a language plus rules which govern correct use of the language.
If you want to learn more about CFGs, I recommend the [Wikipedia page](https://en.wikipedia.org/wiki/Context-free_grammar) as a starting point.
If you're an engineer and want to see the code that turns JSON Schema into a CFG, check out [the outlines code here](https://github.com/outlines-dev/outlines).
Once this is defined, we have our dynamic definition of valid at sample time.
By running this validity test during autoregressive inference, we have knowledge of which tokens in the vocabulary could be valid and simply set the probability of all other tokens to 0 before the final sampling.

Note that this trick requires that you have access to the full logits (the full probability distribution over the token vocabulary) the LLM generates.
This means you need to be using a model API that supports this, or you need to be running your model on your own infrastructure.
Read the original paper [Efficient Guided Generation for Large Language Model](https://arxiv.org/pdf/2307.09702) and have a look at the [outlines](https://outlines-dev.github.io/outlines/) project if you're self-hosting a model in which you'd like to make use of structured outputs.
Recently, OpenAI implemented the same ideas and announced [support for structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) directly in their API.
Hopefully we'll get similar features through other AI model providers soon.

## Conclusion
With structured output, it's possible to guarantee the output format of generated text tokens from a large language model. However, that certainly doesn't guarantee the output is correct. There is some evidence that adding structure could help performance for some tasks. For example, this result on the [GSM8K Benchmark](https://blog.dottxt.co/performance-gsm8k.html). What it certainly does do is help developers use LLMs as components to build robust systems.