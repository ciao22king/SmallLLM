MADE WITH HELP OF MISTRAL VIBE 

# SmallLLM

**La tua AI Locale che gira anche su PC del 2011**

SmallLLM è un modello linguistico leggero progettato per funzionare su hardware vecchio (PC del 2011 con RAM e CPU limitate). Include un'interfaccia web completa simile a ChatGPT, un'API REST compatibile con Byok, e una CLI completa.

## 🌟 Caratteristiche

- **🤖 Modello Tiny Transformer**: 2-4 layer transformer con configurazioni personalizzabili
- **💾 Lazy Loading**: Parametri caricati on-demand per risparmiare memoria
- **🌐 API REST Completa**: Compatibile con Byok e altri servizi
- **💬 Interfaccia Web**: UI moderna simile a ChatGPT
- **⚡ CLI Completa**: 10+ comandi per l'uso da terminale
- **🔒 100% Locale**: Tutti i dati rimangono sul tuo computer
- **🌍 Multilingua**: Supporto per italiano e inglese
- **📦 Quantizzazione 8-bit**: Per ridurre ulteriormente l'uso di memoria

## 📦 Installazione

### Requisiti

- Python 3.6 o successivo
- NumPy 1.16.0 o successivo
- Flask 2.0.0 o successivo (per l'API)

```bash
pip install numpy Flask
```

### Installazione del Pacchetto

```bash
# Clona il repository
git clone https://github.com/ciao22king/SmallLLM.git
cd SmallLLM

# Installa in modalità sviluppo
pip install -e .

# Oppure installa le dipendenze manualmente
pip install -r requirements.txt
```

## 🚀 Avvio Rapido

### 1. Avvia l'API Server

```bash
# Avvia il server API
python -m smallllm.api

# Oppure usa l'entry point
smallllm-api
```

### 2. Avvia l'Interfaccia Web

```bash
# Metodo 1: Usa il server Python integrato
cd web
python run_server.py

# Metodo 2: Usa qualsiasi server web
cd web
python -m http.server 8000
```

### 3. Apri il Browser

Visita `http://localhost:8080` (o la porta che hai specificato) nel tuo browser.

### 4. Configura l'API nell'Interfaccia Web

Nella pagina web:
1. Clicca sull'icona ⚙️ (Impostazioni)
2. Imposta l'URL API su `http://localhost:5000`
3. Clicca "Testa Connessione"
4. Salva le impostazioni

## 🌐 Usare con Byok

1. **Avvia SmallLLM API**:
   ```bash
   python -m smallllm.api --host 0.0.0.0 --port 5000
   ```

2. **In Byok**:
   - Aggiungi un modello personalizzato
   - URL Base: `http://localhost:5000/v1`
   - Nome modello: `smallllm:tiny`, `smallllm:small`, o `smallllm:medium`

3. **Usa il modello** come qualsiasi altro modello in Byok!

## 💬 Interfaccia Web

L'interfaccia web offre:

- **Chat moderna** simile a ChatGPT
- **Cronologia delle conversazioni** con possibilità di eliminare
- **Impostazioni personalizzabili** (tema, URL API, ecc.)
- **Controlli rapidi** per temperatura e max tokens
- **Esempi predefiniti** per iniziare
- **Copia negli appunti** delle risposte
- **Responsive design** per mobile e desktop

### Comandi Tastiera

| Scorciatoia | Azione |
|------------|--------|
| `Ctrl/Cmd + ,` | Apri impostazioni |
| `Ctrl/Cmd + N` | Nuova chat |
| `Enter` | Invia messaggio |
| `Shift + Enter` | Nuova linea |
| `Escape` | Chiudi menu a tendina |

## 🎯 CLI Comandi

### Genera Testo

```bash
smallllm generate --prompt "Ciao come stai?" --max-tokens 50 --config tiny
```

### Chat

```bash
smallllm chat --prompt "Raccontami una barzelletta" --temperature 0.9
```

### Modalità Interattiva

```bash
smallllm interactive
```

### Addestramento (dimostrativo)

```bash
smallllm train --text "Testo per addestramento" --epochs 1 --config tiny
```

### Informazioni Modello

```bash
smallllm info --config small
```

### Memoria

```bash
smallllm memory --config tiny
```

### Comandi Disponibili

```bash
smallllm --help
```

## 🔌 API REST

### Endpoint Standard

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/` | Informazioni API |
| GET | `/api/health` | Health check |
| GET | `/api/info` | Informazioni modello |
| GET | `/api/models` | Lista modelli disponibili |
| GET | `/api/memory` | Uso memoria |
| POST | `/api/generate` | Genera testo |
| POST | `/api/chat` | Chat con AI |
| POST | `/api/tokenize` | Tokenizza testo |
| POST | `/api/detokenize` | Detokenizza testo |
| POST | `/api/batch-generate` | Generazione batch |

### Endpoint Compatibili con Byok

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Chat completions |
| POST | `/v1/completions` | Completions |
| GET | `/v1/models` | Lista modelli |
| GET | `/v1/models/<nome>` | Info modello |

### Esempio Richiesta API

```bash
# Genera testo
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ciao mondo", "max_tokens": 50, "temperature": 0.7}'

# Chat
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Come va?", "max_tokens": 50}'

# Byok-style request
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "smallllm:small", "messages": [{"role": "user", "content": "Ciao!"}], "max_tokens": 50}'
```

## ⚙️ Configurazioni

SmallLLM offre tre configurazioni predefinite:

| Config | Vocab | Embed | Hidden | Layers | Heads | Memoria (32-bit) | Memoria (8-bit) |
|--------|-------|--------|--------|--------|-------|------------------|------------------|
| tiny   | 1024  | 32     | 64     | 2      | 2     | ~0.76 MB         | ~0.19 MB         |
| small  | 2048  | 64     | 128    | 3      | 4     | ~3.79 MB         | ~0.95 MB         |
| medium | 4096  | 128    | 256    | 4      | 8     | ~28 MB           | ~7 MB            |

### Configurazione Personalizzata

```bash
# Crea una configurazione personalizzata
smallllm config --vocab-size 2048 --embedding-dim 64 --hidden-dim 128 --num-layers 3 --save-path custom.json

# Usa la configurazione personalizzata
smallllm generate --prompt "Test" --config-path custom.json
```

## 📊 Python API

```python
from smallllm import SmallLLM, Tokenizer, ModelConfig

# Crea modello con configurazione predefinita
model = SmallLLM()

# Genera testo
result = model.generate("Ciao come stai?", max_new_tokens=50, temperature=0.7)
print(result)

# Chat
response = model.chat("Raccontami una barzelletta")
print(response)

# Usa configurazione personalizzata
config = ModelConfig(
    vocab_size=2048,
    embedding_dim=64,
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
)
model = SmallLLM(config)

# Tokenizza
tokenizer = Tokenizer(vocab_size=2048)
token_ids = tokenizer.encode("Ciao mondo")
text = tokenizer.decode(token_ids)
```

## 💡 Suggerimenti per Hardware Vecchio

### Per PC del 2011 (2-4GB RAM)

1. **Usa la configurazione `tiny`**:
   ```bash
   smallllm generate --config tiny
   ```

2. **Attiva la quantizzazione 8-bit**:
   ```python
   model = SmallLLM()
   model.quantize()  # Quantizza i pesi a 8-bit
   ```

3. **Limita i token**:
   ```bash
   smallllm generate --max-tokens 30
   ```

4. **Usa temperature più basse** per risposte più deterministiche (più veloci):
   ```bash
   smallllm generate --temperature 0.3
   ```

### Per Hardware Moderno

- Usa `small` o `medium` per migliore qualità
- Aumenta `max_new_tokens` per risposte più lunghe
- Usa temperature più alte (0.7-1.0) per risposte più creative

## 🎨 Personalizzazione

### Temi

L'interfaccia web supporta tre temi:
- **Chiaro** (Light)
- **Scuro** (Dark)
- **Sistema** (System) - segue le preferenze del sistema

### Modelli Personalizzati

Puoi creare e addestrare modelli personalizzati:

```bash
# Addestra su un file di testo
smallllm train --file data.txt --epochs 3 --config tiny --save-path my_model

# Carica il modello addestrato
smallllm load --path my_model

# Usa il modello caricato
smallllm generate --prompt "Test" --model-path my_model
```

## 📁 Struttura del Progetto

```
SmallLLM/
├── smallllm/
│   ├── __init__.py      # Package init
│   ├── config.py        # Configurazioni modello
│   ├── tokenizer.py     # Tokenizer leggero
│   ├── model.py         # Modello LLM
│   ├── cli.py           # CLI
│   └── api.py           # API REST
├── web/
│   ├── index.html       # Interfaccia web
│   ├── style.css        # Stili
│   ├── app.js           # Logica JavaScript
│   └── run_server.py    # Server web
├── tests/
│   ├── test_tokenizer.py
│   ├── test_config.py
│   ├── test_model.py
│   └── test_cli.py
├── README.md
├── requirements.txt
├── setup.py
└── api_example.py
```

## 🤝 Contribuire

I contributi sono benvenuti! Sentiti libero di:

1. Reportare bug
2. Suggerire nuove funzionalità
3. Inviare pull request
4. Migliorare la documentazione

## 📜 Licenza

MIT License - vedi [LICENSE](LICENSE) per i dettagli.

---

**Goditi la tua AI locale su qualsiasi hardware!** 🎉

> "SmallLLM: Intelligenza artificiale per tutti, anche su hardware vecchio"
