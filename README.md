## [nomic-ai/modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base)

|`max_position_embeddings`|`hidden_size`|`num_hidden_layers`|`pooling`
|-:|-:|-:|-:|
|`8192`|`768`|`22`|`mean`

```4d
var $en; $fr : 4D.Vector
var $AIClient : cs.AIKit.OpenAI
var $cosineSimilarity : Real
$AIClient:=cs.AIKit.OpenAI.new()

$AIClient.baseURL:="http://127.0.0.1:8081/v1"  // onnx-genai

$inputs:=[\
"Il m'a posé un lapin hier soir."; \
"Il n'est pas venu à notre rendez-vous."; \
"Ich verstehe nur Bahnhof."; \
"Das ist mir völlig unklar und verwirrend."; \
"In bocca al lupo per il tuo esame!"; \
"Ti auguro tanto successo per la prova."; \
"Me estás tomando el pelo."; \
"Creo que me estás engañando con una broma."\
]

$batch:=$AIClient.embeddings.create($inputs)

$fr1:=$batch.embeddings[0].embedding
$fr2:=$batch.embeddings[1].embedding
$de1:=$batch.embeddings[2].embedding
$de2:=$batch.embeddings[3].embedding
$it1:=$batch.embeddings[4].embedding
$it2:=$batch.embeddings[5].embedding
$es1:=$batch.embeddings[6].embedding
$es2:=$batch.embeddings[7].embedding

$cosineSimilarity1:=$fr1.cosineSimilarity($fr2)
$cosineSimilarity2:=$de1.cosineSimilarity($de2)
$cosineSimilarity3:=$it1.cosineSimilarity($it2)
$cosineSimilarity4:=$es1.cosineSimilarity($es2)
```

##### Cosine similarity from example code above:

||llama.cpp `Q8_0`|ONNX Runtime `Int8`|
|-|:-|:-|
|🇫🇷|`0.6489139717033`|`0.6113182768442`|
|🇩🇪|`0.6023781437155`|`0.6296701786738`|
|🇮🇹|`0.6358381136145`|`0.6474611891482`|
|🇪🇸|`0.728039126897`|`0.7302156058768`|
