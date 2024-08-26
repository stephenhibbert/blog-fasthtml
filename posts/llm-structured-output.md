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

Building robust software with Large Language Models (LLMs) presents unique challenges.
In this post, I'll explain practically how some of these challenges can be mitigated with structured outputs.

First, consider a simple chatbot application:

<Image alt="the cat sat on the" src="/public/images/llm-structured-outputs/chatbot.png" width={500} height={200} />

The user 
- ① types a question which is
- ② encoded and input to the LLM, then 
- ③ generates an output which is decoded and rendered on the UI to be
- ④ read by the user. 

LLMs are text token generators. We give them an input sequence and then predict a probability distribution over the next item in the sequence.

Let's take a real example, consider the following string:

```json
The cat sat on the
```

Using this as the input to a language model generator, we get the following output distribution for the next token (showing the top 10 probabilities):

<Image alt="the cat sat on the" src="/public/images/llm-structured-outputs/0.png" width={1000} height={500} />

A common way to sample the next token from this distribution is called **greedy decoding**.
This is simply selecting the token with the highest probability. 
In the case above, the token with the highest probability is `mat`.

In our chatbot, we have a UI to directly display the LLM output to a human, so we free to sample any token from the distribution without constraint. 
The text may look funny, contain factual errors or even undesirable content, but it won't break our software. 

Next, imagine we are building a new application: an automated email triage system for a vet.
Customer emails need to be routed to the proper department, and so the type of animal needs to be extracted from the email.

<Image alt="the cat sat on the" src="/public/images/llm-structured-outputs/vet.png" width={500} height={200} />

A customer...
- ① sends an email, which is encoded and input to the LLM, then 
- ② should output JSON with the animal species so it can
- ③ be stored in a database to enable the downstream applications

```json
{
  "animal": "cat"
}
```

Note that the LLM generator is near the start of our system. 
And we require a contract between components so software downstream of the LLM can make correct assumptions about the output data. In our case, the output should contain a JSON document, with a required `animal` key that has a string value.

## Level 1 - Prompting

A naive starting point is to give instructions, or even plead with the LLM in natural language:

```json
What's the main animal mentioned in the email? Output in JSON.
Please please please output valid JSON, my career depends on it!
```

This is better than no instructions at all and will work some of the time.
However, we will frequently get structural errors like spurious arrays or misspelt keys:

```json
{
  "animals": ["cat"]
}
```

Developers can try their best to fix JSON errors on the fly with libraries like [json_repair](https://github.com/mangiucugna/json_repair).
This is fundamentally fragile making it hard to ship features using this method in a production codebase.

## Level 2 - Function calling (tool use)

We can do better by being explicit about the fact we want to generate JSON and defining the required fields, their data types, and any validation rules upfront.
An easy way to do this in python is with the [Pydantic](https://docs.pydantic.dev/latest/) library.

First, we create a data model:

```python
from pydantic import BaseModel

class Animal(BaseModel):
    animal: str

print(Animal.model_json_schema())
```

... and use the built in `model_json_schema()` method to create a [JSON Schema](https://json-schema.org/) document:

```
{'properties': {'animal': {'title': 'Animal', 'type': 'string'}}, 'required': ['animal'], 'title': 'Animal', 'type': 'object'}
```

Now we have a formal definition of *valid* written in code we can validate the output from the LLM.
But how can we input this to help guide the LLM to generate the JSON what we want?

Models have been optimised to learn how to take JSON Schema and return JSON to support function calling.
We can take advantage of this feature to ask for structured output.
This is made easy with the [Instructor](https://python.useinstructor.com/) library, by simply supplying the Pydantic model in the `response_model`:

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

This improves our chances greatly, and now we can validate and retry the request in-case we get malformed output.
Instructor helps save developers time by abstracting the data modelling, prompts, retries and validation logic. But, we still don't get any guarantees that the model will be compliant with the JSON Schema we asked for 100% of the time.

## Level 3 - Structured Output

We can do even better if we pull some tricks in the way tokens are sampled from the language model.
This is a family of techniques called constrained decoding.
Remember that the fundamental challenge is that the definition of a valid token according to a JSON Schema is a function of the position in the output sequence.
As an example, imagine we're midway through a generation:

```
{"ani
```

Our LLM predicts this distribution for the next token:

<Image alt="1" src="/public/images/llm-structured-outputs/1.png" width={1000} height={500} />

Notice that only a subset of possible tokens are valid JSON according to the JSON Schema above.
Invalid tokens are greyed out.

We take the highest probability token and append it to the prior sequence,

```
{"animal
```

...and then generate again...

<Image alt="2" src="/public/images/llm-structured-outputs/2.png" width={1000} height={500} />

...append the most probable next token...

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
The definition of *valid* changes based on the values of previous tokens.
All other tokens (shown greyed out) were masked when sampling from the distribution.
When creating these visuals, I manually identified the invalid tokens.
We need a programmatic and dynamic way to define what is a valid next token to implement constrained decoding.

One way this can be achieved is by using a context-free grammar (CFG).
This is a formal way to specify a language plus rules which govern correct use of the language.
A JSON grammar can be written down in [ENBF](https://www.wikiwand.com/en/articles/Extended_Backus%E2%80%93Naur_form). 

```tex
?start: value

?value: object
| array
| UNESCAPED_STRING
| SIGNED_NUMBER      -> number
| "true"             -> true
| "false"            -> false
| "null"             -> null

array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : UNESCAPED_STRING ":" value

%import common.UNESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS

%ignore WS
```

See the [Lark](https://lark-parser.readthedocs.io/en/latest/json_tutorial.html) docs for more details about how this is constructed.

The last piece in the puzzle is the code to connect the CFG parser to the output of an LLM during token sampling. [Outlines](https://outlines-dev.github.io/outlines/) project that makes this easy if you're self-hosting a model. Note that this trick requires that you have access to the full logits (the full probability distribution over the token vocabulary) the LLM generates. This means you need to be using a model API that supports this, or you need to be running your model on your own infrastructure.

Read the original paper from the [.txt](https://dottxt.co/) team [Efficient Guided Generation for Large Language Model](https://arxiv.org/pdf/2307.09702) or checkout their [blog](https://blog.dottxt.co/) for more details.

Recently, OpenAI implemented the same ideas and announced [support for structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) directly in their API.
Hopefully we'll get similar features through other AI model providers soon.

## Conclusion
With structured output, it's possible to guarantee the output format of generated text tokens from a large language model. However, that certainly doesn't guarantee the output is correct. What it certainly does do is help developers use LLMs as components to build robust systems.
