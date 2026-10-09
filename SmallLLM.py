#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MINI AI 1.0
===========
Rete neurale per cifre + Assistente completo + Modello linguistico neurale.
Python puro (solo librerie standard). Compatibile con Python 2.7+ / 3.x.

Funzionalita':
- Riconoscimento cifre 0-9 con rete neurale (35 -> 20 -> 12 -> 10)
- Modello linguistico neurale incrementale (embedding + ReLU + softmax)
- Assistente con risposte complete (Wikipedia + DuckDuckGo + dizionario)
- Assistente coding (Stack Overflow + sintesi)
- Memoria conversazionale persistente
- Comandi slash: /help /esc /think /fast /model /verbose /history /clear
  /corpus /train /dict /wiki /code /lang /status /save /calc /time /date
  /version /about /tips /note /notes /clearnotes /explain /quiz /export
  /import /random /joke /quote /personality /reset /translate /followup
  /confidence /sources /keywords /brief /deep /simplify /related /recap
- Modalita' pensiero (/think) con passaggi interni visibili
- Personalita' configurabili, note, calcolatrice, quiz, jokes, quotes
"""

from __future__ import print_function, division

import random
import math
import json
import re
import os
import sys
import time
import datetime
from collections import Counter

try:
    from urllib.request import urlopen, Request
    from urllib.parse import quote
except ImportError:
    from urllib2 import urlopen, Request
    from urllib import quote

random.seed(42)

CONFIG_FILE           = "mini_ai_config.json"
WEIGHTS_FILE          = "mini_ai_weights.json"
DICTIONARY_CACHE_FILE = "mini_ai_dictionary.json"
WORD_FREQ_FILE        = "mini_ai_word_freq.json"
CORPUS_FILE           = "mini_ai_corpus.txt"
LM_WEIGHTS_FILE       = "mini_ai_lm_weights.json"
MEMORY_FILE           = "mini_ai_memory.json"
VOCAB_FILE            = "mini_ai_vocab.json"
NOTES_FILE            = "mini_ai_notes.json"
PERSONALITY_FILE      = "mini_ai_personality.json"

USER_AGENT = "MiniAI-1.0-Educational/1.0"
VERSION    = "1.0"

MAX_VOCAB     = 500
EMBEDDING_DIM = 32
HIDDEN_LM     = 64
CONTEXT_SIZE  = 3

JOKES = [
    "Perche' i programmatori preferiscono il buio? Perche' la luce attira i bug.",
    "Un SQL entra in un bar, si avvicina a due tavoli e chiede: posso unirvi?",
    "Ci sono 10 tipi di persone: quelle che capiscono il binario e quelle che no.",
    "Come si chiama un programmatore che non commenta il codice? Disoccupato.",
    "Un byte entra in un bar e ordina una birra. Il barista: ancora una? Il byte: no, mi basta un bit.",
    "Perche' Java e' triste? Perche' ha troppe classi.",
    "Un programmatore va a letto e mette due bicchieri sul comodino: uno pieno d'acqua, uno vuoto. Perche'? Se ha sete beve, se non ha sete no.",
    "Come fai a riconoscere un programmatore estroverso? Guarda le sue scarpe: se sono allacciate, e' introverso.",
    "Cosa dice un bit all'altro? Ci vediamo al byte.",
    "Il codice non ha bug, ha solo funzionalita' non documentate."
]

QUOTES = [
    "Il miglior codice e' quello che non devi scrivere. - Anonimo",
    "Parla poco e mostra il codice. - Linus Torvalds",
    "Semplice e' meglio di complesso. - Zen di Python",
    "Leggibile e' meglio di compatto. - Zen di Python",
    "Il debug e' due volte piu' difficile dello scrivere codice. - Brian Kernighan",
    "Ogni grande developer che conosci ha scritto un sacco di codice brutto. - Anonimo",
    "Non e' un bug, e' una feature non documentata.",
    "Prima risolvi il problema, poi scrivi il codice. - John Johnson",
    "L'esperienza e' il nome che diamo ai nostri errori. - Oscar Wilde",
    "La programmazione non e' conoscere tutto, e' sapere dove cercare."
]

PERSONALITIES = {
    "default":   {"prefix": "", "suffix": "", "brief": False},
    "friendly":  {"prefix": "Ciao! ", "suffix": " Spero ti sia utile!", "brief": False},
    "formal":    {"prefix": "Buongiorno. ", "suffix": " Resto a disposizione.", "brief": False},
    "brief":     {"prefix": "", "suffix": "", "brief": True},
    "sarcastic": {"prefix": "Oh, guarda un po'. ", "suffix": " Prego, non c'e' di che.", "brief": False},
    "teacher":   {"prefix": "Vediamo insieme. ", "suffix": " Ricorda: capire e' meglio che memorizzare.", "brief": False},
    "scientific": {"prefix": "Analisi: ", "suffix": " Dati basati su fonti pubbliche.", "brief": False}
}

def _is_py2():
    return sys.version_info[0] == 2

def _input(prompt):
    return raw_input(prompt) if _is_py2() else input(prompt)

def _http_get_json(url, timeout=8):
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        resp = urlopen(req, timeout=timeout)
        raw = resp.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception as e:
        _dprint("(Errore di rete: %s)" % e)
        return None

def _split_into_chunks(text, max_len=450):
    if len(text) <= max_len:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_len:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks

def translate_text(text, source_lang, target_lang):
    if not text or source_lang == target_lang:
        return text
    parts = []
    for chunk in _split_into_chunks(text):
        url = ("https://api.mymemory.translated.net/get?q=%s&langpair=%s|%s"
               % (quote(chunk), source_lang, target_lang))
        data = _http_get_json(url)
        if data and data.get("responseData"):
            parts.append(data["responseData"].get("translatedText", chunk))
        else:
            parts.append(chunk)
    return " ".join(parts)

class AppState(object):
    def __init__(self):
        self.thinking      = False
        self.verbose       = True
        self.followup      = True
        self.confidence    = True
        self.brief_mode    = False
        self.deep_mode     = False
        self.personality   = "default"
        self.config        = {}
        self.word_freq     = {}
        self.dict_cache    = {}
        self.memory        = []
        self.vocab         = {}
        self.lm            = None
        self.notes         = []
        self.last_answer   = ""
        self.last_query    = ""
        self.last_keywords = []

STATE = AppState()

def _dprint(*args, **kwargs):
    if STATE.verbose or STATE.thinking:
        print(*args, **kwargs)

def _tprint(*args, **kwargs):
    if STATE.thinking:
        print(*args, **kwargs)

LANGUAGES = [("it", "Italiano"), ("en", "English"), ("es", "Espanol"),
             ("fr", "Francais"), ("de", "Deutsch")]

UI_STRINGS = {
    "title": "MINI AI 1.0 - Rete neurale + Assistente + Modello linguistico",
    "training": "Nessun modello salvato trovato: allenamento della rete neurale in corso...",
    "training_done": "Allenamento completato!",
    "weights_loaded": "Pesi caricati da disco - nessun allenamento necessario!",
    "menu_header": "--- MENU ---",
    "menu_1": "1) Mostra le cifre di riferimento",
    "menu_2": "2) Test automatico con rumore (rete neurale locale)",
    "menu_3": "3) Disegna una cifra e falla riconoscere",
    "menu_4": "4) Modalita' Assistente (richiede internet)",
    "menu_5": "5) Assistente per il coding (Stack Overflow + IA)",
    "menu_6": "6) Esci",
    "choice_prompt": "Scelta: ",
    "invalid_choice": "Scelta non valida.",
    "bye": "Ciao!",
    "draw_intro": "Disegna la tua cifra: 7 righe da 5 caratteri (0 o 1). Esempio: 01110",
    "draw_row": "Riga %d: ",
    "draw_invalid": "Input non valido, riprova (5 caratteri, solo 0 e 1).",
    "your_digit": "La tua cifra:",
    "network_thinks": "La rete pensa che sia: %d (confidenza %.1f%%)",
    "assistant_intro": "Scrivi una domanda, oppure /help per i comandi. /esc per tornare al menu.",
    "assistant_prompt": "Chiedi: ",
    "coding_intro": "Poni una domanda di programmazione, oppure /help per i comandi. /esc per tornare.",
    "coding_prompt": "Codice> ",
    "keywords_extracted": "Parole chiave individuate: %s",
    "searching_wikipedia": "Cerco su Wikipedia...",
    "searching_ddg": "Cerco su DuckDuckGo...",
    "searching_so": "Cerco su Stack Overflow...",
    "searching_dictionary": "Cerco i significati delle parole chiave nel dizionario...",
    "no_result_at_all": "Non ho trovato nulla di utile. Prova a riformulare la domanda in modo piu' semplice.",
    "source": "Fonte: %s",
    "definition_label": "Definizione",
    "no_dictionary_result": "Nessuna voce di dizionario trovata.",
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(config):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)
    except Exception as e:
        print("(Impossibile salvare la configurazione: %s)" % e)

def get_ui(key, config):
    lang = config.get("language", "it")
    base = UI_STRINGS[key]
    if lang == "it":
        return base
    cache = config.setdefault("ui_translations", {}).setdefault(lang, {})
    if key in cache:
        return cache[key]
    translated = translate_text(base, "it", lang)
    cache[key] = translated
    save_config(config)
    return translated

def choose_language_first_run():
    print("=" * 60)
    print("Scegli la lingua / Choose language / Elige idioma /")
    print("Choisissez la langue / Waehle die Sprache:")
    for i, (code, name) in enumerate(LANGUAGES, start=1):
        print("%d) %s" % (i, name))
    while True:
        choice = _input("> ").strip()
        for i, (code, name) in enumerate(LANGUAGES, start=1):
            if choice == str(i) or choice.lower() == code:
                return code
        print("Scelta non valida / Invalid choice, riprova / try again")

DIGIT_PATTERNS = {
    0: ["01110","10001","10011","10101","11001","10001","01110"],
    1: ["00100","01100","00100","00100","00100","00100","01110"],
    2: ["01110","10001","00001","00010","00100","01000","11111"],
    3: ["11110","00001","00001","01110","00001","00001","11110"],
    4: ["00010","00110","01010","10010","11111","00010","00010"],
    5: ["11111","10000","11110","00001","00001","10001","01110"],
    6: ["00110","01000","10000","11110","10001","10001","01110"],
    7: ["11111","00001","00010","00100","01000","01000","01000"],
    8: ["01110","10001","10001","01110","10001","10001","01110"],
    9: ["01110","10001","10001","01111","00001","00010","01100"],
}

def pattern_to_vector(rows):
    return [1.0 if ch == "1" else 0.0 for row in rows for ch in row]

def print_digit(vec, width=5, height=7):
    for r in range(height):
        print("".join("#" if vec[r * width + c] > 0.5 else "." for c in range(width)))

def add_noise(vec, flip_prob=0.08):
    return [1.0 - v if random.random() < flip_prob else v for v in vec]

def sigmoid(x):
    if x < -60: return 0.0
    if x > 60: return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_deriv(y):
    return y * (1.0 - y)

class DeepMiniNeuralNetwork(object):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, lr=0.4, momentum=0.7):
        self.n_input, self.n_hidden1 = n_input, n_hidden1
        self.n_hidden2, self.n_output = n_hidden2, n_output
        self.lr, self.momentum = lr, momentum
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(n_hidden1)] for _ in range(n_input)]
        self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(n_hidden2)] for _ in range(n_hidden1)]
        self.w3 = [[random.uniform(-0.5, 0.5) for _ in range(n_output)] for _ in range(n_hidden2)]
        self.b1 = [random.uniform(-0.5, 0.5) for _ in range(n_hidden1)]
        self.b2 = [random.uniform(-0.5, 0.5) for _ in range(n_hidden2)]
        self.b3 = [random.uniform(-0.5, 0.5) for _ in range(n_output)]
        self.vw1 = [[0.0] * n_hidden1 for _ in range(n_input)]
        self.vw2 = [[0.0] * n_hidden2 for _ in range(n_hidden1)]
        self.vw3 = [[0.0] * n_output for _ in range(n_hidden2)]
        self.vb1 = [0.0] * n_hidden1
        self.vb2 = [0.0] * n_hidden2
        self.vb3 = [0.0] * n_output

    def forward(self, inputs):
        h1 = [sigmoid(self.b1[j] + sum(inputs[i] * self.w1[i][j] for i in range(self.n_input))) for j in range(self.n_hidden1)]
        h2 = [sigmoid(self.b2[j] + sum(h1[i] * self.w2[i][j] for i in range(self.n_hidden1))) for j in range(self.n_hidden2)]
        out = [sigmoid(self.b3[j] + sum(h2[i] * self.w3[i][j] for i in range(self.n_hidden2))) for j in range(self.n_output)]
        return h1, h2, out

    def train_step(self, inputs, targets):
        h1, h2, out = self.forward(inputs)
        d_out = [(targets[o] - out[o]) * sigmoid_deriv(out[o]) for o in range(self.n_output)]
        d_h2 = [sum(d_out[o] * self.w3[j][o] for o in range(self.n_output)) * sigmoid_deriv(h2[j]) for j in range(self.n_hidden2)]
        d_h1 = [sum(d_h2[k] * self.w2[j][k] for k in range(self.n_hidden2)) * sigmoid_deriv(h1[j]) for j in range(self.n_hidden1)]
        for j in range(self.n_hidden2):
            for o in range(self.n_output):
                g = self.lr * d_out[o] * h2[j]
                self.vw3[j][o] = self.momentum * self.vw3[j][o] + g
                self.w3[j][o] += self.vw3[j][o]
        for o in range(self.n_output):
            g = self.lr * d_out[o]
            self.vb3[o] = self.momentum * self.vb3[o] + g
            self.b3[o] += self.vb3[o]
        for i in range(self.n_hidden1):
            for j in range(self.n_hidden2):
                g = self.lr * d_h2[j] * h1[i]
                self.vw2[i][j] = self.momentum * self.vw2[i][j] + g
                self.w2[i][j] += self.vw2[i][j]
        for j in range(self.n_hidden2):
            g = self.lr * d_h2[j]
            self.vb2[j] = self.momentum * self.vb2[j] + g
            self.b2[j] += self.vb2[j]
        for i in range(self.n_input):
            for j in range(self.n_hidden1):
                g = self.lr * d_h1[j] * inputs[i]
                self.vw1[i][j] = self.momentum * self.vw1[i][j] + g
                self.w1[i][j] += self.vw1[i][j]
        for j in range(self.n_hidden1):
            g = self.lr * d_h1[j]
            self.vb1[j] = self.momentum * self.vb1[j] + g
            self.b1[j] += self.vb1[j]
        return sum((targets[o] - out[o]) ** 2 for o in range(self.n_output))

    def predict(self, inputs):
        _, _, out = self.forward(inputs)
        best_idx = max(range(len(out)), key=lambda i: out[i])
        return best_idx, out

    def save_weights(self, fp):
        data = {"n_input": self.n_input, "n_hidden1": self.n_hidden1,
                "n_hidden2": self.n_hidden2, "n_output": self.n_output,
                "w1": self.w1, "w2": self.w2, "w3": self.w3,
                "b1": self.b1, "b2": self.b2, "b3": self.b3}
        try:
            with open(fp, "w") as f:
                json.dump(data, f)
            return True
        except Exception as e:
            print("(Impossibile salvare i pesi: %s)" % e)
            return False

    @classmethod
    def load_weights(cls, fp, expected_input, expected_output):
        if not os.path.exists(fp):
            return None
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            if data.get("n_input") != expected_input or data.get("n_output") != expected_output:
                return None
            net = cls(data["n_input"], data["n_hidden1"], data["n_hidden2"], data["n_output"])
            net.w1, net.w2, net.w3 = data["w1"], data["w2"], data["w3"]
            net.b1, net.b2, net.b3 = data["b1"], data["b2"], data["b3"]
            return net
        except Exception as e:
            print("(File di pesi corrotto: %s -> riallenamento)" % e)
            return None

def build_dataset():
    inputs, targets = [], []
    for d in range(10):
        vec = pattern_to_vector(DIGIT_PATTERNS[d])
        target = [0.0] * 10
        target[d] = 1.0
        inputs.append(vec)
        targets.append(target)
    return inputs, targets

def train_network(net, inputs, targets, epochs=1500, verbose=True):
    n = len(inputs)
    for epoch in range(epochs):
        total = 0.0
        order = list(range(n))
        random.shuffle(order)
        for idx in order:
            sample = add_noise(inputs[idx], flip_prob=0.05)
            total += net.train_step(sample, targets[idx])
        if verbose and (epoch % 300 == 0 or epoch == epochs - 1):
            print("  epoca %d/%d - errore totale: %.4f" % (epoch, epochs, total))
    return net

def show_reference_digits():
    for d in range(10):
        print("\nCifra: %d" % d)
        print_digit(pattern_to_vector(DIGIT_PATTERNS[d]))

def test_with_noise(net):
    print("\nTest su cifre con rumore casuale:\n")
    correct = 0
    for d in range(10):
        noisy = add_noise(pattern_to_vector(DIGIT_PATTERNS[d]), flip_prob=0.15)
        pred, out = net.predict(noisy)
        print("Cifra reale: %d | Predetta: %d %s" % (d, pred, "OK" if pred == d else "ERRATO"))
        print_digit(noisy)
        print("Confidenza: %.1f%%\n" % (out[pred] * 100))
        if pred == d:
            correct += 1
    print("Risultato: %d/10 corrette (%.0f%%)" % (correct, 100.0 * correct / 10))

def draw_custom_digit(config):
    print("\n" + get_ui("draw_intro", config))
    rows = []
    for r in range(7):
        while True:
            line = _input(get_ui("draw_row", config) % (r + 1)).strip()
            if len(line) == 5 and all(ch in "01" for ch in line):
                rows.append(line)
                break
            print(get_ui("draw_invalid", config))
    return pattern_to_vector(rows)

class NeuralLanguageModel(object):
    def __init__(self, vocab_size, embed_dim=EMBEDDING_DIM, hidden=HIDDEN_LM):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.emb = [[random.uniform(-0.1, 0.1) for _ in range(embed_dim)] for _ in range(vocab_size)]
        self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(hidden)] for _ in range(CONTEXT_SIZE * embed_dim)]
        self.b1 = [0.0] * hidden
        self.W2 = [[random.uniform(-0.1, 0.1) for _ in range(vocab_size)] for _ in range(hidden)]
        self.b2 = [0.0] * vocab_size
        self.vW1 = [[0.0] * hidden for _ in range(CONTEXT_SIZE * embed_dim)]
        self.vb1 = [0.0] * hidden
        self.vW2 = [[0.0] * vocab_size for _ in range(hidden)]
        self.vb2 = [0.0] * vocab_size
        self.lr = 0.05
        self.momentum = 0.9

    def _softmax(self, logits):
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def _relu(self, x):
        return x if x > 0 else 0.0

    def _relu_deriv(self, x):
        return 1.0 if x > 0 else 0.0

    def forward(self, ctx):
        flat = []
        for idx in ctx:
            if 0 <= idx < len(self.emb):
                flat.extend(self.emb[idx])
            else:
                flat.extend([0.0] * self.embed_dim)
        h = [self.b1[j] + sum(flat[i] * self.W1[i][j] for i in range(len(flat))) for j in range(self.hidden)]
        h_relu = [self._relu(x) for x in h]
        out = [self.b2[k] + sum(h_relu[j] * self.W2[j][k] for j in range(self.hidden)) for k in range(self.vocab_size)]
        return h_relu, self._softmax(out)

    def train_step(self, ctx, target_idx):
        h_relu, probs = self.forward(ctx)
        if target_idx >= self.vocab_size:
            return 0.0
        d_out = [probs[k] - (1.0 if k == target_idx else 0.0) for k in range(self.vocab_size)]
        grad_W2 = [[h_relu[j] * d_out[k] for k in range(self.vocab_size)] for j in range(self.hidden)]
        grad_b2 = d_out[:]
        d_h = [sum(d_out[k] * self.W2[j][k] for k in range(self.vocab_size)) * self._relu_deriv(h_relu[j]) for j in range(self.hidden)]
        flat = []
        for idx in ctx:
            if 0 <= idx < len(self.emb):
                flat.extend(self.emb[idx])
            else:
                flat.extend([0.0] * self.embed_dim)
        grad_W1 = [[flat[i] * d_h[j] for j in range(self.hidden)] for i in range(len(flat))]
        grad_b1 = d_h[:]
        for i in range(len(flat)):
            for j in range(self.hidden):
                self.vW1[i][j] = self.momentum * self.vW1[i][j] + self.lr * grad_W1[i][j]
                self.W1[i][j] -= self.vW1[i][j]
        for j in range(self.hidden):
            self.vb1[j] = self.momentum * self.vb1[j] + self.lr * grad_b1[j]
            self.b1[j] -= self.vb1[j]
        for j in range(self.hidden):
            for k in range(self.vocab_size):
                self.vW2[j][k] = self.momentum * self.vW2[j][k] + self.lr * grad_W2[j][k]
                self.W2[j][k] -= self.vW2[j][k]
        for k in range(self.vocab_size):
            self.vb2[k] = self.momentum * self.vb2[k] + self.lr * grad_b2[k]
            self.b2[k] -= self.vb2[k]
        for pos, idx in enumerate(ctx):
            if not (0 <= idx < len(self.emb)):
                continue
            for e in range(self.embed_dim):
                g = self.lr * sum(d_h[j] * self.W1[pos * self.embed_dim + e][j] for j in range(self.hidden))
                self.emb[idx][e] -= g
        return -math.log(probs[target_idx] + 1e-10)

    def predict_next(self, ctx, temperature=0.8):
        _, probs = self.forward(ctx)
        if temperature < 1e-6:
            return max(range(len(probs)), key=lambda i: probs[i])
        logits = [math.log(p + 1e-10) / temperature for p in probs]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        s = sum(exps)
        p = [e / s for e in exps]
        r = random.random()
        cum = 0.0
        for i, v in enumerate(p):
            cum += v
            if r < cum:
                return i
        return len(p) - 1

    def save(self, fp):
        data = {"vocab_size": self.vocab_size, "embed_dim": self.embed_dim,
                "hidden": self.hidden, "emb": self.emb, "W1": self.W1,
                "b1": self.b1, "W2": self.W2, "b2": self.b2}
        try:
            with open(fp, "w") as f:
                json.dump(data, f)
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, fp):
        if not os.path.exists(fp):
            return None
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            lm = cls(data["vocab_size"], data["embed_dim"], data["hidden"])
            lm.emb = data["emb"]; lm.W1 = data["W1"]; lm.b1 = data["b1"]
            lm.W2 = data["W2"]; lm.b2 = data["b2"]
            return lm
        except Exception:
            return None

def load_vocab():
    if os.path.exists(VOCAB_FILE):
        try:
            with open(VOCAB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"word2idx": {}, "idx2word": [], "counts": {}}

def save_vocab(v):
    try:
        with open(VOCAB_FILE, "w") as f:
            json.dump(v, f)
    except Exception:
        pass

def add_word_to_vocab(word, v):
    if word in v["word2idx"]:
        v["counts"][word] = v["counts"].get(word, 0) + 1
        return
    if len(v["idx2word"]) >= MAX_VOCAB:
        min_word = min(v["counts"], key=v["counts"].get)
        if v["counts"][min_word] >= v["counts"].get(word, 0):
            return
        idx = v["word2idx"].pop(min_word)
        v["idx2word"][idx] = word
        v["word2idx"][word] = idx
        v["counts"][word] = v["counts"].pop(min_word, 0) + 1
    else:
        idx = len(v["idx2word"])
        v["word2idx"][word] = idx
        v["idx2word"].append(word)
        v["counts"][word] = v["counts"].get(word, 0) + 1

def _tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\sàèéìòùáéíóúñü]", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if len(t) > 0]

def load_corpus():
    if os.path.exists(CORPUS_FILE):
        try:
            with open(CORPUS_FILE, "r") as f:
                return f.read()
        except Exception:
            pass
    return ""

def append_corpus(text, max_chars=200000):
    text = (text or "").strip()
    if not text:
        return
    try:
        with open(CORPUS_FILE, "a") as f:
            f.write(text + "\n")
        content = load_corpus()
        if len(content) > max_chars:
            with open(CORPUS_FILE, "w") as f:
                f.write(content[-max_chars:])
    except Exception:
        pass

def train_lm_on_corpus(lm, vocab, epochs=3):
    corpus = load_corpus()
    tokens = _tokenize(corpus)
    if len(tokens) < CONTEXT_SIZE + 1:
        return 0.0
    indices = []
    for t in tokens:
        if t not in vocab["word2idx"]:
            add_word_to_vocab(t, vocab)
        if t in vocab["word2idx"]:
            indices.append(vocab["word2idx"][t])
    if len(indices) < CONTEXT_SIZE + 1:
        return 0.0
    if lm.vocab_size != len(vocab["idx2word"]):
        new_size = max(100, len(vocab["idx2word"]))
        new_lm = NeuralLanguageModel(new_size)
        old = min(lm.vocab_size, new_size)
        for i in range(old):
            new_lm.emb[i] = lm.emb[i][:]
        for i in range(min(len(lm.W1), len(new_lm.W1))):
            new_lm.W1[i] = lm.W1[i][:]
        for i in range(min(len(lm.W2), len(new_lm.W2))):
            new_lm.W2[i] = (lm.W2[i] + [0.0] * new_size)[:new_size]
        lm = new_lm
        STATE.lm = lm
    total_loss = 0.0
    for _ in range(epochs):
        for i in range(len(indices) - CONTEXT_SIZE):
            ctx = indices[i:i + CONTEXT_SIZE]
            total_loss += lm.train_step(ctx, indices[i + CONTEXT_SIZE])
    lm.save(LM_WEIGHTS_FILE)
    save_vocab(vocab)
    return total_loss

def generate_lm_text(lm, vocab, seed_text, n_words=30, temperature=0.7):
    if lm is None or not vocab["idx2word"]:
        return ""
    tokens = _tokenize(seed_text)
    if len(tokens) < CONTEXT_SIZE:
        ctx_words = (["<PAD>"] * CONTEXT_SIZE + tokens)[-CONTEXT_SIZE:]
    else:
        ctx_words = tokens[-CONTEXT_SIZE:]
    ctx = [vocab["word2idx"].get(t, 0) for t in ctx_words]
    out = list(ctx_words)
    for _ in range(n_words):
        nxt = lm.predict_next(ctx, temperature)
        w = vocab["idx2word"][nxt] if nxt < len(vocab["idx2word"]) else "<UNK>"
        out.append(w)
        ctx = ctx[1:] + [nxt]
    return " ".join(out)

def _median(values):
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2.0 if n % 2 == 0 else s[mid]

def load_word_freq():
    if os.path.exists(WORD_FREQ_FILE):
        try:
            with open(WORD_FREQ_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"__total_queries__": 0}

def save_word_freq(wf):
    try:
        with open(WORD_FREQ_FILE, "w") as f:
            json.dump(wf, f)
    except Exception:
        pass

def extract_keywords_auto(text, wf):
    tokens = _tokenize(text)
    if not tokens:
        return []
    total = wf.get("__total_queries__", 0)
    median_len = _median([len(t) for t in tokens])
    keywords = []
    for t in tokens:
        if len(t) <= 2:
            continue
        freq = wf.get(t, 0)
        if total >= 5 and freq / float(total) > 0.3:
            continue
        if total < 5 and len(t) < median_len:
            continue
        keywords.append(t)
    if not keywords:
        keywords = [t for t in tokens if len(t) > 2]
    for t in set(tokens):
        wf[t] = wf.get(t, 0) + 1
    wf["__total_queries__"] = total + 1
    save_word_freq(wf)
    return keywords

def build_search_queries(original, wf):
    queries = [original.strip()]
    kws = extract_keywords_auto(original, wf)
    if kws:
        kq = " ".join(kws)
        if kq.lower() != original.strip().lower():
            queries.append(kq)
    return queries, kws

def _wikipedia_lookup(query, lang):
    url = ("https://%s.wikipedia.org/w/api.php?action=opensearch&search=%s"
           "&limit=1&namespace=0&format=json" % (lang, quote(query)))
    data = _http_get_json(url)
    if not data or len(data) < 2 or not data[1]:
        return None
    title = data[1][0]
    summary_url = ("https://%s.wikipedia.org/api/rest_v1/page/summary/%s"
                   % (lang, quote(title.replace(" ", "_"))))
    summary = _http_get_json(summary_url)
    if not summary or "extract" not in summary or not summary["extract"]:
        return None
    return {"title": summary.get("title", title), "extract": summary.get("extract", ""),
            "url": summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "lang": lang}

def ask_wikipedia(query, wf):
    variants, _ = build_search_queries(query, wf)
    for lang in ("it", "en"):
        for q in variants:
            r = _wikipedia_lookup(q, lang)
            if r:
                return r
    return None

def _duckduckgo_lookup(query):
    url = ("https://api.duckduckgo.com/?q=%s&format=json&no_html=1"
           "&skip_disambig=1" % quote(query))
    data = _http_get_json(url)
    if not data:
        return None
    abstract = data.get("AbstractText") or ""
    if abstract:
        return {"text": abstract,
                "source": data.get("AbstractSource", "DuckDuckGo"),
                "url": data.get("AbstractURL", "")}
    ans = data.get("Answer") or ""
    if ans:
        return {"text": ans, "source": "DuckDuckGo", "url": ""}
    return None

def ask_duckduckgo(query, wf):
    variants, _ = build_search_queries(query, wf)
    for q in variants:
        r = _duckduckgo_lookup(q)
        if r:
            return r
    return None

def ask_stackoverflow(query, wf):
    variants, _ = build_search_queries(query, wf)
    for q in variants:
        url = ("https://api.stackexchange.com/2.3/search/advanced?order=desc"
               "&sort=relevance&q=%s&site=stackoverflow" % quote(q))
        data = _http_get_json(url)
        if data and data.get("items"):
            item = data["items"][0]
            qid = item["question_id"]
            ans_url = ("https://api.stackexchange.com/2.3/questions/%d/answers"
                       "?order=desc&sort=votes&site=stackoverflow&filter=withbody" % qid)
            ad = _http_get_json(ans_url)
            if ad and ad.get("items"):
                best = ad["items"][0]
                body = best.get("body", "")
                body = re.sub(r"<[^>]+>", " ", body)
                body = re.sub(r"\s+", " ", body).strip()
                if body:
                    return {"text": body, "source": "Stack Overflow",
                            "url": item.get("link", "")}
    return None

def ask_wikipedia_random(lang="it"):
    url = "https://%s.wikipedia.org/api/rest_v1/page/random/summary" % lang
    data = _http_get_json(url)
    if data and data.get("extract"):
        return {"title": data.get("title", "?"),
                "extract": data.get("extract", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "lang": lang}
    return None

def load_dictionary_cache():
    if os.path.exists(DICTIONARY_CACHE_FILE):
        try:
            with open(DICTIONARY_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_dictionary_cache(c):
    try:
        with open(DICTIONARY_CACHE_FILE, "w") as f:
            json.dump(c, f)
    except Exception:
        pass

def _dictionary_lookup(word, lang):
    url = "https://api.dictionaryapi.dev/api/v2/entries/%s/%s" % (lang, quote(word))
    data = _http_get_json(url)
    if not data or not isinstance(data, list):
        return None
    meanings = []
    for entry in data:
        for m in entry.get("meanings", []):
            pos = m.get("partOfSpeech", "")
            for d in m.get("definitions", [])[:2]:
                definition = d.get("definition", "")
                if definition:
                    meanings.append((pos, definition))
    return meanings if meanings else None

def ask_dictionary(word, lang, cache):
    key = "%s:%s" % (lang, word.lower())
    if key in cache:
        return cache[key]
    result = _dictionary_lookup(word, lang)
    cache[key] = result
    save_dictionary_cache(cache)
    return result

def _clean_sentence(s):
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?":
        s += "."
    return s

def _extract_relevant_sentences(text, keywords, max_sentences=2):
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    kw_set = set(k.lower() for k in keywords)
    scored = []
    for i, s in enumerate(sents):
        s_clean = s.strip()
        if len(s_clean) < 25:
            continue
        words = set(_tokenize(s_clean))
        overlap = len(words & kw_set)
        pos_bonus = 1.0 / (1 + i * 0.25)
        scored.append((overlap * 2 + pos_bonus, s_clean))
    if not scored:
        return []
    scored.sort(key=lambda x: -x[0])
    return [_clean_sentence(s) for _, s in scored[:max_sentences]]

def summarize_text(text, max_sentences=3):
    if not text:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) <= max_sentences:
        return text
    words = _tokenize(text)
    freq = Counter(w for w in words if len(w) > 2)
    if not freq:
        return text[:500]
    scores = []
    for s in sents:
        ws = _tokenize(s)
        if not ws:
            scores.append(0)
            continue
        scores.append(sum(freq.get(w, 0) for w in ws) / len(ws))
    ranked = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
    chosen = sorted(ranked[:max_sentences])
    return " ".join(sents[i] for i in chosen)

def simplify_text(text, max_sentences=4):
    if not text:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", text)
    simple = []
    for s in sents[:max_sentences]:
        s = _clean_sentence(s)
        s = s.replace(";", ",")
        s = re.sub(r"\s+", " ", s)
        if len(s) > 200:
            parts = s.split(", ")
            s = ". ".join(parts[:3]) + "."
        simple.append(s)
    return " ".join(simple)

def _build_main_paragraph(wiki, ddg, target_lang):
    frags = []
    if wiki and wiki.get("extract"):
        text = wiki["extract"]
        if wiki.get("lang") != target_lang:
            text = translate_text(text, wiki["lang"], target_lang)
        sents = re.split(r"(?<=[.!?])\s+", text)
        n = 5 if STATE.deep_mode else 3
        main = " ".join(_clean_sentence(s) for s in sents[:n] if s.strip())
        if main:
            frags.append(main)
    if ddg and ddg.get("text"):
        ddg_text = ddg["text"]
        if target_lang != "en":
            ddg_text = translate_text(ddg_text, "autodetect", target_lang)
        if not frags or ddg_text[:80].lower() not in frags[0].lower():
            frags.append(_clean_sentence(ddg_text))
    return " ".join(frags)

def _compute_confidence(wiki, ddg, dict_meanings, keywords):
    score = 0.0
    if wiki and wiki.get("extract"):
        score += 0.5
        if len(wiki["extract"]) > 300:
            score += 0.1
    if ddg and ddg.get("text"):
        score += 0.2
    if dict_meanings:
        score += 0.1
    if keywords:
        score += 0.1
    return min(1.0, score)

def _apply_personality(text, personality):
    p = PERSONALITIES.get(personality, PERSONALITIES["default"])
    if not text:
        return text
    if p["brief"]:
        return text
    return p["prefix"] + text + p["suffix"]

def synthesize_answer(query, keywords, wiki, ddg, dict_meanings, lm, vocab, config):
    lang = config.get("language", "it")
    sections = []

    main = _build_main_paragraph(wiki, ddg, lang)
    if main:
        if STATE.brief_mode:
            main = simplify_text(main, 2)
        sections.append(main)

    if wiki and wiki.get("extract"):
        raw = wiki["extract"]
        if wiki.get("lang") != lang:
            raw = translate_text(raw, wiki["lang"], lang)
        n = 4 if STATE.deep_mode else 2
        key_sents = _extract_relevant_sentences(raw, keywords, max_sentences=n)
        if key_sents and main:
            key_sents = [s for s in key_sents if s[:50] not in main]
        if key_sents:
            sections.append("Punti chiave: " + " ".join(key_sents))

    if dict_meanings and len(keywords) <= 2:
        defs = []
        for pos, d in dict_meanings[:2]:
            d_clean = _clean_sentence(d)
            defs.append("%s (%s)" % (d_clean, pos) if pos else d_clean)
        if defs:
            sections.append("Definizione: " + " ".join(defs))

    if STATE.deep_mode and sections and lm is not None:
        seed = main if main else query
        generated = generate_lm_text(lm, vocab, seed_text=seed,
                                     n_words=40, temperature=0.7)
        if generated:
            gen_words = generated.split()
            seed_words = _tokenize(seed)[-CONTEXT_SIZE:]
            if len(gen_words) > len(seed_words):
                generated = " ".join(gen_words[len(seed_words):])
            if lang != "it" and generated:
                generated = translate_text(generated, "autodetect", lang)
            if generated.strip():
                sections.append("[Approfondimento generato - SPERIMENTALE]\n" + generated)

    if not sections:
        return None
    answer = "\n\n".join(sections)
    if STATE.confidence:
        conf = _compute_confidence(wiki, ddg, dict_meanings, keywords)
        answer += "\n\n[Confidenza: %.0f%%]" % (conf * 100)
    answer = _apply_personality(answer, STATE.personality)
    return answer

def _related_topics(keywords, wiki):
    related = []
    if wiki and wiki.get("extract"):
        text = wiki["extract"]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        words = Counter()
        for s in sentences[:5]:
            for w in _tokenize(s):
                if len(w) > 4 and w not in keywords:
                    words[w] += 1
        related = [w for w, _ in words.most_common(5)]
    return related

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_memory(m):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(m[-50:], f)
    except Exception:
        pass

def load_notes():
    if os.path.exists(NOTES_FILE):
        try:
            with open(NOTES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_notes(n):
    try:
        with open(NOTES_FILE, "w") as f:
            json.dump(n, f)
    except Exception:
        pass

def load_personality():
    if os.path.exists(PERSONALITY_FILE):
        try:
            with open(PERSONALITY_FILE, "r") as f:
                data = json.load(f)
                return data.get("personality", "default")
        except Exception:
            pass
    return "default"

def save_personality(p):
    try:
        with open(PERSONALITY_FILE, "w") as f:
            json.dump({"personality": p}, f)
    except Exception:
        pass

def safe_calculate(expr):
    allowed = set("0123456789+-*/()., %")
    if not all(c in allowed for c in expr):
        return None, "Espressione contiene caratteri non ammessi."
    expr = expr.replace(",", ".").replace("%", "/100.0*")
    if expr.endswith("*"):
        expr = expr[:-1]
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return result, None
    except Exception as e:
        return None, "Errore di calcolo: %s" % e

HELP_TEXT = """
Comandi disponibili:

  --- Sessione ---
  /help                    Mostra questo elenco
  /esc                     Torna al menu principale
  /about                   Informazioni sul programma
  /version                 Versione e dettagli tecnici
  /tips                    Suggerimenti d'uso
  /status                  Stato generale del sistema
  /save                    Salva tutto su disco
  /reset all               Cancella tutti i dati salvati
  /export <file>           Esporta la memoria conversazionale
  /import <file>           Importa la memoria conversazionale

  --- Ragionamento ---
  /think                   Attiva la modalita' pensiero
  /fast                    Disattiva la modalita' pensiero
  /model think             Alias di /think
  /model fast              Alias di /fast
  /model status            Stato del modello neurale
  /deep                    Attiva approfondimento esteso
  /brief                   Attiva modalita' breve
  /normal                  Torna alla modalita' normale
  /confidence              Mostra/nascondi la confidenza
  /followup                Mostra/nascondi i suggerimenti di follow-up

  --- Visualizzazione ---
  /verbose on|off          Dettagli completi on/off
  /sources on|off          Fonti in calce on/off
  /keywords on|off         Parole chiave on/off

  --- Contenuti ---
  /dict <parola>           Cerca una parola nel dizionario
  /wiki <query>            Cerca su Wikipedia
  /random                  Articolo casuale da Wikipedia
  /explain <argomento>     Spiegazione semplificata
  /simplify                Riscrive l'ultima risposta in modo semplice
  /recap                   Riassume l'ultima risposta
  /related                 Argomenti correlati all'ultima risposta
  /quiz <argomento>        Genera una domanda di verifica
  /translate <lang> <txt>  Traduce un testo (it, en, es, fr, de)

  --- Coding ---
  /code <query>            Passa alla modalita' coding con quella query

  --- Personale ---
  /note <testo>            Salva una nota
  /notes                   Elenca le note salvate
  /clearnotes              Cancella tutte le note
  /personality <modo>      Cambia personalita' (default, friendly, formal,
                           brief, sarcastic, teacher, scientific)

  --- Utility ---
  /calc <espressione>      Calcolatrice (es. /calc (2+3)*4)
  /time                    Ora corrente
  /date                    Data corrente
  /joke                    Barzelletta casuale
  /quote                   Citazione casuale
  /lang <codice>           Cambia lingua (it, en, es, fr, de)
  /history                 Ultime interazioni
  /clear                   Cancella la memoria conversazionale
  /corpus                  Statistiche sul corpus locale
  /train                   Forza un ciclo di addestramento
"""

def handle_slash_command(line, config, word_freq, dict_cache, lm, vocab, memory):
    line = line.strip()
    if not line.startswith("/"):
        return False, "none", None

    parts = line[1:].split()
    if not parts:
        return True, "continue", None
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd == "help":
        print(HELP_TEXT)
        return True, "continue", None

    if cmd in ("esc", "exit", "quit", "menu"):
        return True, "menu", None

    if cmd == "think":
        STATE.thinking = True
        print("(Modalita' pensiero attivata. Usa /fast per disattivarla.)")
        return True, "continue", None

    if cmd == "fast":
        STATE.thinking = False
        print("(Modalita' pensiero disattivata.)")
        return True, "continue", None

    if cmd == "deep":
        STATE.deep_mode = True
        STATE.brief_mode = False
        print("(Modalita' approfondita attivata.)")
        return True, "continue", None

    if cmd == "brief":
        STATE.brief_mode = True
        STATE.deep_mode = False
        print("(Modalita' breve attivata.)")
        return True, "continue", None

    if cmd == "normal":
        STATE.brief_mode = False
        STATE.deep_mode = False
        print("(Modalita' normale attivata.)")
        return True, "continue", None

    if cmd == "confidence":
        STATE.confidence = not STATE.confidence
        print("(Confidenza: %s)" % ("ON" if STATE.confidence else "OFF"))
        return True, "continue", None

    if cmd == "followup":
        STATE.followup = not STATE.followup
        print("(Follow-up: %s)" % ("ON" if STATE.followup else "OFF"))
        return True, "continue", None

    if cmd == "model":
        sub = args[0].lower() if args else ""
        if sub == "think":
            STATE.thinking = True
            print("(Modalita' pensiero attivata.)")
        elif sub == "fast":
            STATE.thinking = False
            print("(Modalita' pensiero disattivata.)")
        elif sub == "status":
            if lm is None:
                print("Modello linguistico: non inizializzato.")
            else:
                print("Modello linguistico neurale:")
                print("  Vocabolario: %d parole" % lm.vocab_size)
                print("  Embedding: %d dimensioni" % lm.embed_dim)
                print("  Layer nascosto: %d neuroni" % lm.hidden)
                print("  Contesto: %d parole" % CONTEXT_SIZE)
                params = lm.vocab_size * lm.embed_dim
                params += CONTEXT_SIZE * lm.embed_dim * lm.hidden
                params += lm.hidden * lm.vocab_size
                print("  Parametri totali (circa): %d" % params)
        else:
            print("Uso: /model think | /model fast | /model status")
        return True, "continue", None

    if cmd == "verbose":
        if not args:
            print("Verbose: %s" % ("on" if STATE.verbose else "off"))
        elif args[0].lower() in ("on", "1", "true"):
            STATE.verbose = True
            print("(Verbose attivato.)")
        elif args[0].lower() in ("off", "0", "false"):
            STATE.verbose = False
            print("(Verbose disattivato.)")
        else:
            print("Uso: /verbose on|off")
        return True, "continue", None

    if cmd == "sources":
        if not args:
            print("Fonti: %s" % ("on" if STATE.verbose else "off"))
        else:
            STATE.verbose = args[0].lower() in ("on", "1", "true")
            print("(Fonti: %s)" % ("ON" if STATE.verbose else "OFF"))
        return True, "continue", None

    if cmd == "keywords":
        if not args:
            print("Keywords: %s" % ("on" if STATE.verbose else "off"))
        else:
            STATE.verbose = args[0].lower() in ("on", "1", "true")
            print("(Keywords: %s)" % ("ON" if STATE.verbose else "OFF"))
        return True, "continue", None

    if cmd == "history":
        if not memory:
            print("(Nessuna interazione in memoria.)")
        else:
            for i, h in enumerate(memory[-10:], 1):
                q = h.get("query", "")[:80]
                a = (h.get("answer") or "")[:120].replace("\n", " ")
                print("  %d. %s" % (i, q))
                if a:
                    print("     -> %s..." % a)
        return True, "continue", None

    if cmd == "clear":
        memory[:] = []
        save_memory(memory)
        print("(Memoria conversazionale cancellata.)")
        return True, "continue", None

    if cmd == "corpus":
        corpus = load_corpus()
        words = corpus.split()
        print("Corpus locale:")
        print("  Caratteri: %d" % len(corpus))
        print("  Parole: %d" % len(words))
        print("  Vocabolario attivo: %d parole" % len(vocab.get("idx2word", [])))
        return True, "continue", None

    if cmd == "train":
        print("(Addestramento del modello linguistico in corso...)")
        loss = train_lm_on_corpus(lm, vocab, epochs=5)
        print("(Addestramento completato. Loss totale: %.2f)" % loss)
        return True, "continue", None

    if cmd in ("dict", "define"):
        if not args:
            print("Uso: /dict <parola>")
            return True, "continue", None
        word = args[0]
        lang = config.get("language", "it")
        meanings = ask_dictionary(word, lang, dict_cache)
        if meanings:
            print("Definizioni di '%s':" % word)
            for pos, d in meanings[:4]:
                print("  %s%s" % ("(%s) " % pos if pos else "", d))
        else:
            print("(Nessuna definizione trovata per '%s'.)" % word)
        return True, "continue", None

    if cmd == "wiki":
        if not args:
            print("Uso: /wiki <query>")
            return True, "continue", None
        q = " ".join(args)
        print("(Cerco '%s' su Wikipedia...)" % q)
        r = ask_wikipedia(q, word_freq)
        if r:
            text = r["extract"]
            if r["lang"] != config.get("language", "it"):
                text = translate_text(text, r["lang"], config.get("language", "it"))
            print("\n[%s]\n%s" % (r["title"], summarize_text(text, 4)))
            if r["url"]:
                print(get_ui("source", config) % r["url"])
            append_corpus(r["extract"])
        else:
            print("(Nessun risultato.)")
        return True, "continue", None

    if cmd == "random":
        lang = config.get("language", "it")
        if lang not in ("it", "en"):
            lang = "en"
        print("(Cerco un articolo casuale da Wikipedia...)")
        r = ask_wikipedia_random(lang)
        if r:
            text = r["extract"]
            print("\n[%s]" % r["title"])
            print(summarize_text(text, 3))
            if r["url"]:
                print(get_ui("source", config) % r["url"])
            append_corpus(r["extract"])
        else:
            print("(Nessun risultato.)")
        return True, "continue", None

    if cmd == "explain":
        if not args:
            print("Uso: /explain <argomento>")
            return True, "continue", None
        q = " ".join(args)
        print("(Preparo una spiegazione semplice di '%s'...)" % q)
        r = ask_wikipedia(q, word_freq)
        if r:
            text = r["extract"]
            if r["lang"] != config.get("language", "it"):
                text = translate_text(text, r["lang"], config.get("language", "it"))
            simple = simplify_text(text, 3)
            print("\n[Spiegazione semplice]")
            print(simple)
            if r["url"]:
                print(get_ui("source", config) % r["url"])
            append_corpus(r["extract"])
        else:
            print("(Nessun risultato da semplificare.)")
        return True, "continue", None

    if cmd == "simplify":
        if not STATE.last_answer:
            print("(Nessuna risposta precedente da semplificare.)")
        else:
            print("\n[Riscrittura semplice]")
            print(simplify_text(STATE.last_answer, 4))
        return True, "continue", None

    if cmd == "recap":
        if not STATE.last_answer:
            print("(Nessuna risposta precedente da riassumere.)")
        else:
            print("\n[Riassunto]")
            print(summarize_text(STATE.last_answer, 3))
        return True, "continue", None

    if cmd == "related":
        if not STATE.last_answer:
            print("(Nessuna risposta precedente.)")
        else:
            rel = _related_topics(STATE.last_keywords, {"extract": STATE.last_answer})
            if rel:
                print("(Argomenti correlati: %s)" % ", ".join(rel))
            else:
                print("(Nessun argomento correlato trovato.)")
        return True, "continue", None

    if cmd == "quiz":
        if not args:
            print("Uso: /quiz <argomento>")
            return True, "continue", None
        q = " ".join(args)
        print("(Genero un quiz su '%s'...)" % q)
        r = ask_wikipedia(q, word_freq)
        if not r:
            print("(Nessun materiale per il quiz.)")
            return True, "continue", None
        text = r["extract"]
        if r["lang"] != config.get("language", "it"):
            text = translate_text(text, r["lang"], config.get("language", "it"))
        sents = re.split(r"(?<=[.!?])\s+", text)
        if len(sents) < 2:
            print("(Testo troppo breve per un quiz.)")
            return True, "continue", None
        target_sent = sents[0]
        kws = [w for w in _tokenize(target_sent) if len(w) > 4]
        if not kws:
            print("(Impossibile generare una domanda.)")
            return True, "continue", None
        correct = kws[0]
        blanked = target_sent.replace(correct, "_____", 1)
        wrong_pool = []
        for s in sents[1:]:
            for w in _tokenize(s):
                if len(w) > 4 and w != correct and w not in wrong_pool:
                    wrong_pool.append(w)
        if len(wrong_pool) < 3:
            for w in _tokenize(text):
                if len(w) > 4 and w != correct and w not in wrong_pool:
                    wrong_pool.append(w)
                if len(wrong_pool) >= 3:
                    break
        options = [correct] + wrong_pool[:3]
        random.shuffle(options)
        print("\n[Quiz] Completa la frase:")
        print("  %s" % blanked)
        for i, o in enumerate(options, 1):
            print("  %d) %s" % (i, o))
        print("  (Risposta corretta: %s)" % correct)
        return True, "continue", None

    if cmd == "translate":
        if len(args) < 2:
            print("Uso: /translate <lang> <testo>")
            return True, "continue", None
        target = args[0].lower()
        valid = [c for c, _ in LANGUAGES]
        if target not in valid:
            print("Lingua non valida. Valide: %s" % ", ".join(valid))
            return True, "continue", None
        text = " ".join(args[1:])
        src = config.get("language", "it")
        result = translate_text(text, src, target)
        print("[%s -> %s] %s" % (src, target, result))
        return True, "continue", None

    if cmd == "code":
        if not args:
            print("(Passo alla modalita' coding. Scrivi la tua domanda.)")
            return True, "code", ""
        return True, "code", " ".join(args)

    if cmd == "lang":
        if not args:
            print("Lingua attuale: %s" % config.get("language", "it"))
            return True, "continue", None
        code = args[0].lower()
        valid = [c for c, _ in LANGUAGES]
        if code in valid:
            config["language"] = code
            save_config(config)
            print("(Lingua cambiata in: %s)" % code)
        else:
            print("(Lingua non valida. Valide: %s)" % ", ".join(valid))
        return True, "continue", None

    if cmd == "status":
        print("=== STATO DEL SISTEMA ===")
        print("Versione: %s" % VERSION)
        print("Lingua: %s" % config.get("language", "it"))
        print("Pensiero: %s" % ("ON" if STATE.thinking else "OFF"))
        print("Verbose: %s" % ("ON" if STATE.verbose else "OFF"))
        print("Deep: %s" % ("ON" if STATE.deep_mode else "OFF"))
        print("Brief: %s" % ("ON" if STATE.brief_mode else "OFF"))
        print("Confidenza: %s" % ("ON" if STATE.confidence else "OFF"))
        print("Follow-up: %s" % ("ON" if STATE.followup else "OFF"))
        print("Personalita': %s" % STATE.personality)
        print("Interazioni in memoria: %d" % len(memory))
        corpus = load_corpus()
        print("Corpus: %d parole" % len(corpus.split()))
        print("Vocabolario: %d parole" % len(vocab.get("idx2word", [])))
        if lm:
            print("Modello LM: %d parole, %d dim embedding" %
                  (lm.vocab_size, lm.embed_dim))
        return True, "continue", None

    if cmd == "save":
        save_config(config)
        save_memory(memory)
        save_vocab(vocab)
        save_word_freq(word_freq)
        save_dictionary_cache(dict_cache)
        save_notes(STATE.notes)
        save_personality(STATE.personality)
        if lm:
            lm.save(LM_WEIGHTS_FILE)
        print("(Stato salvato su disco.)")
        return True, "continue", None

    if cmd == "note":
        if not args:
            print("Uso: /note <testo>")
            return True, "continue", None
        text = " ".join(args)
        entry = {"text": text, "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
        STATE.notes.append(entry)
        save_notes(STATE.notes)
        print("(Nota salvata: %s)" % text)
        return True, "continue", None

    if cmd == "notes":
        if not STATE.notes:
            print("(Nessuna nota salvata.)")
        else:
            print("Note salvate (%d):" % len(STATE.notes))
            for i, n in enumerate(STATE.notes, 1):
                print("  %d. [%s] %s" % (i, n.get("time", "?"), n.get("text", "")))
        return True, "continue", None

    if cmd == "clearnotes":
        STATE.notes = []
        save_notes(STATE.notes)
        print("(Tutte le note cancellate.)")
        return True, "continue", None

    if cmd == "personality":
        if not args:
            print("Personalita' attuale: %s" % STATE.personality)
            print("Disponibili: %s" % ", ".join(PERSONALITIES.keys()))
            return True, "continue", None
        p = args[0].lower()
        if p in PERSONALITIES:
            STATE.personality = p
            save_personality(p)
            print("(Personalita' impostata: %s)" % p)
        else:
            print("Personalita' non valida. Disponibili: %s" % ", ".join(PERSONALITIES.keys()))
        return True, "continue", None

    if cmd == "calc":
        if not args:
            print("Uso: /calc <espressione>")
            return True, "continue", None
        expr = " ".join(args)
        result, err = safe_calculate(expr)
        if err:
            print("(%s)" % err)
        else:
            print("%s = %s" % (expr, result))
        return True, "continue", None

    if cmd == "time":
        print("Ora corrente: %s" % datetime.datetime.now().strftime("%H:%M:%S"))
        return True, "continue", None

    if cmd == "date":
        print("Data corrente: %s" % datetime.datetime.now().strftime("%d/%m/%Y"))
        return True, "continue", None

    if cmd == "joke":
        print(random.choice(JOKES))
        return True, "continue", None

    if cmd == "quote":
        print(random.choice(QUOTES))
        return True, "continue", None

    if cmd == "version":
        print("MINI AI versione %s" % VERSION)
        print("Python %s" % sys.version.split()[0])
        print("Rete neurale cifre: 35 -> 20 -> 12 -> 10")
        print("Modello LM: embedding=%d, hidden=%d, contesto=%d" %
              (EMBEDDING_DIM, HIDDEN_LM, CONTEXT_SIZE))
        print("Vocabolario massimo: %d parole" % MAX_VOCAB)
        return True, "continue", None

    if cmd == "about":
        print("MINI AI %s" % VERSION)
        print("Assistente locale scritto in Python puro (solo librerie standard).")
        print("Combina una rete neurale per cifre, un modello linguistico")
        print("neurale incrementale e un assistente che sintetizza risposte")
        print("da Wikipedia, DuckDuckGo, Stack Overflow e dizionario.")
        print("Funziona su PC datati dal 2011 in poi.")
        print("Nessuna installazione richiesta, nessuna chiave API.")
        return True, "continue", None

    if cmd == "tips":
        print("Suggerimenti:")
        print("  - Usa /think per vedere come ragiona l'assistente.")
        print("  - /deep allunga le risposte, /brief le accorcia.")
        print("  - /personality cambia il tono dell'assistente.")
        print("  - /note salva appunti tra le sessioni.")
        print("  - /random scopre un articolo di Wikipedia a caso.")
        print("  - /explain semplifica un argomento complesso.")
        print("  - /quiz verifica cosa hai imparato.")
        print("  - /train rafforza il modello linguistico locale.")
        return True, "continue", None

    if cmd == "reset":
        if not args or args[0].lower() != "all":
            print("Uso: /reset all")
            return True, "continue", None
        files = [CONFIG_FILE, WEIGHTS_FILE, DICTIONARY_CACHE_FILE, WORD_FREQ_FILE,
                 CORPUS_FILE, LM_WEIGHTS_FILE, MEMORY_FILE, VOCAB_FILE,
                 NOTES_FILE, PERSONALITY_FILE]
        for f in files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        print("(Tutti i dati salvati sono stati cancellati. Riavvia per ricominciare.)")
        return True, "menu", None

    if cmd == "export":
        if not args:
            print("Uso: /export <file>")
            return True, "continue", None
        fp = args[0]
        try:
            with open(fp, "w") as f:
                json.dump(memory, f)
            print("(Memoria esportata in '%s': %d interazioni.)" % (fp, len(memory)))
        except Exception as e:
            print("(Errore durante l'esportazione: %s)" % e)
        return True, "continue", None

    if cmd == "import":
        if not args:
            print("Uso: /import <file>")
            return True, "continue", None
        fp = args[0]
        if not os.path.exists(fp):
            print("(File non trovato: %s)" % fp)
            return True, "continue", None
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                memory.extend(data)
                save_memory(memory)
                print("(Importate %d interazioni da '%s'.)" % (len(data), fp))
            else:
                print("(Formato non valido.)")
        except Exception as e:
            print("(Errore durante l'importazione: %s)" % e)
        return True, "continue", None

    print("(Comando sconosciuto: %s. Usa /help per l'elenco.)" % cmd)
    return True, "continue", None

def _suggest_followup(keywords, wiki):
    related = _related_topics(keywords, wiki)
    if not related:
        return None
    return "Vuoi che approfondisca: %s?" % ", ".join(related[:3])

def assistant_mode(config, word_freq, dict_cache, lm, vocab, memory):
    print("\n--- Assistente ---")
    print(get_ui("assistant_intro", config))

    while True:
        query = _input(get_ui("assistant_prompt", config)).strip()
        if not query:
            continue

        if query.startswith("/"):
            handled, action, payload = handle_slash_command(
                query, config, word_freq, dict_cache, lm, vocab, memory)
            if action == "menu":
                return
            if action == "code":
                coding_assistant_mode(config, word_freq, lm, vocab, dict_cache, memory)
                continue
            if action == "continue":
                continue

        if query.lower() in ("fine", "exit", "quit", "end"):
            return

        if STATE.thinking:
            print("\n[THINK] Analizzo la domanda...")
        _, keywords = build_search_queries(query, word_freq)
        if keywords:
            if STATE.verbose or STATE.thinking:
                print(get_ui("keywords_extracted", config) % ", ".join(keywords))
            _tprint("[THINK] Parole chiave scelte: %s" % ", ".join(keywords))

        if STATE.thinking:
            print("[THINK] Cerco informazioni sulle fonti disponibili...")
        wiki = ask_wikipedia(query, word_freq)
        ddg = ask_duckduckgo(query, word_freq)

        dict_meanings = []
        for kw in (keywords[:2] if keywords else []):
            m = ask_dictionary(kw, config.get("language", "it"), dict_cache)
            if m:
                dict_meanings.extend(m)
                for _, definition in m[:3]:
                    append_corpus(definition)

        if STATE.thinking:
            if wiki:
                print("[THINK] Wikipedia: trovato '%s' (%d caratteri)"
                      % (wiki.get("title", "?"), len(wiki.get("extract", ""))))
            else:
                print("[THINK] Wikipedia: nessun risultato.")
            if ddg:
                print("[THINK] DuckDuckGo: %d caratteri." % len(ddg.get("text", "")))
            else:
                print("[THINK] DuckDuckGo: nessun risultato.")
            if dict_meanings:
                print("[THINK] Dizionario: %d definizioni." % len(dict_meanings))
            else:
                print("[THINK] Dizionario: nessuna definizione.")

        if wiki and wiki.get("extract"):
            append_corpus(wiki["extract"])
        if ddg and ddg.get("text"):
            append_corpus(ddg["text"])

        corpus = load_corpus()
        if len(corpus.split()) >= 50:
            if STATE.thinking:
                print("[THINK] Addestro il modello linguistico sul corpus aggiornato...")
            train_lm_on_corpus(lm, vocab, epochs=2)

        if STATE.thinking:
            print("[THINK] Sintetizzo la risposta...")
        answer = synthesize_answer(query, keywords, wiki, ddg,
                                   dict_meanings, lm, vocab, config)

        print("\n" + "=" * 60)
        if answer:
            print(answer)
            STATE.last_answer = answer
            STATE.last_query = query
            STATE.last_keywords = keywords
        else:
            print(get_ui("no_result_at_all", config))
        print("=" * 60)

        if STATE.verbose and ((wiki and wiki.get("url")) or (ddg and ddg.get("url"))):
            print("")
            if wiki and wiki.get("url"):
                print(get_ui("source", config) % wiki["url"])
            if ddg and ddg.get("url"):
                print(get_ui("source", config) % ddg["url"])

        if STATE.followup and answer:
            sug = _suggest_followup(keywords, wiki)
            if sug:
                print("\n" + sug)

        memory.append({"query": query, "answer": answer})
        save_memory(memory)
        print("")

def coding_assistant_mode(config, word_freq, lm, vocab, dict_cache, memory):
    print("\n--- Assistente Coding ---")
    print(get_ui("coding_intro", config))

    pending = None
    while True:
        if pending is not None:
            query = pending
            pending = None
        else:
            query = _input(get_ui("coding_prompt", config)).strip()
        if not query:
            continue

        if query.startswith("/"):
            handled, action, payload = handle_slash_command(
                query, config, word_freq, dict_cache, lm, vocab, memory)
            if action == "menu":
                return
            if action == "code":
                if payload:
                    pending = payload
                continue
            if action == "continue":
                continue

        if query.lower() in ("fine", "exit", "quit", "end"):
            return

        if STATE.thinking:
            print("\n[THINK] Analizzo la domanda di programmazione...")
        _, keywords = build_search_queries(query, word_freq)
        if keywords and (STATE.verbose or STATE.thinking):
            print(get_ui("keywords_extracted", config) % ", ".join(keywords))

        if STATE.thinking:
            print("[THINK] Interrogo Stack Overflow...")
        so = ask_stackoverflow(query, word_freq)
        if so and so.get("text"):
            summary = summarize_text(so["text"], max_sentences=5)
            if STATE.thinking:
                print("[THINK] Trovata risposta su Stack Overflow (%d caratteri)."
                      % len(so["text"]))
            print("\n[Risposta da Stack Overflow]")
            print(summary)
            if STATE.verbose and so.get("url"):
                print(get_ui("source", config) % so["url"])
            append_corpus(so["text"])
        else:
            if STATE.thinking:
                print("[THINK] Stack Overflow: nessun risultato diretto.")
            print("(Nessuna risposta diretta su Stack Overflow.)")

        if STATE.thinking:
            print("[THINK] Genero una possibile soluzione con il modello...")
        corpus = load_corpus()
        if len(corpus.split()) >= 50:
            train_lm_on_corpus(lm, vocab, epochs=2)
            generated = generate_lm_text(lm, vocab, seed_text=query,
                                         n_words=40, temperature=0.6)
            if generated:
                if config.get("language", "it") != "it":
                    generated = translate_text(generated, "autodetect",
                                               config.get("language", "it"))
                print("\n[Approfondimento generato - SPERIMENTALE]")
                print(_apply_personality(generated, STATE.personality))
        else:
            print("(Il modello linguistico non ha ancora abbastanza dati.)")

        memory.append({"query": "[code] " + query,
                       "answer": so.get("text") if so else None})
        save_memory(memory)
        print("")

def main():
    config = load_config()
    if "language" not in config:
        config["language"] = choose_language_first_run()
        save_config(config)

    STATE.config = config
    STATE.personality = load_personality()
    STATE.notes = load_notes()

    print("=" * 60)
    print(" " + get_ui("title", config))
    print("=" * 60)
    print(" Suggerimento: in modalita' assistente usa /help per i comandi.")

    net = DeepMiniNeuralNetwork.load_weights(WEIGHTS_FILE, 35, 10)
    if net is not None:
        print("\n" + get_ui("weights_loaded", config))
    else:
        print("\n" + get_ui("training", config) + "\n")
        inputs, targets = build_dataset()
        net = DeepMiniNeuralNetwork(35, 20, 12, 10, 0.4, 0.7)
        train_network(net, inputs, targets, epochs=1500)
        print("\n" + get_ui("training_done", config))
        net.save_weights(WEIGHTS_FILE)

    vocab = load_vocab()
    lm = NeuralLanguageModel.load(LM_WEIGHTS_FILE)
    if lm is None:
        vs = max(100, len(vocab["idx2word"]))
        lm = NeuralLanguageModel(vs)
        print("(Modello linguistico inizializzato: vocab=%d)" % vs)
    STATE.lm = lm
    STATE.vocab = vocab
    STATE.word_freq = load_word_freq()
    STATE.dict_cache = load_dictionary_cache()
    STATE.memory = load_memory()

    word_freq = STATE.word_freq
    dict_cache = STATE.dict_cache
    memory = STATE.memory

    while True:
        print("\n" + get_ui("menu_header", config))
        print(get_ui("menu_1", config))
        print(get_ui("menu_2", config))
        print(get_ui("menu_3", config))
        print(get_ui("menu_4", config))
        print(get_ui("menu_5", config))
        print(get_ui("menu_6", config))
        choice = _input(get_ui("choice_prompt", config)).strip()

        if choice == "1":
            show_reference_digits()
        elif choice == "2":
            test_with_noise(net)
        elif choice == "3":
            vec = draw_custom_digit(config)
            print("\n" + get_ui("your_digit", config))
            print_digit(vec)
            pred, out = net.predict(vec)
            print("\n>>> " + get_ui("network_thinks", config) % (pred, out[pred] * 100))
        elif choice == "4":
            assistant_mode(config, word_freq, dict_cache, lm, vocab, memory)
        elif choice == "5":
            coding_assistant_mode(config, word_freq, lm, vocab, dict_cache, memory)
        elif choice == "6":
            print(get_ui("bye", config))
            save_config(config)
            save_memory(memory)
            save_vocab(vocab)
            save_word_freq(word_freq)
            save_dictionary_cache(dict_cache)
            save_notes(STATE.notes)
            save_personality(STATE.personality)
            if lm:
                lm.save(LM_WEIGHTS_FILE)
            break
        else:
            print(get_ui("invalid_choice", config))

if __name__ == "__main__":
    main()