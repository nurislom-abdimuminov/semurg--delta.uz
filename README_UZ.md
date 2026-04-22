# Semurg' Delta V2

> **O'zbek tili uchun yengil neyron tarmoq arxitekturasi**  
> Original arxitektura — BERT sozlamasi emas

**Muallif:** Nurislom Abdumuminov | 2025–2026  
**Email:** exampil180@gmail.com  
**GitHub:** [nurislom-abdimuminov](https://github.com/nurislom-abdimuminov)

---

## Semurg' Delta nima?

Semurg' Delta V2 — O'zbek tili uchun maxsus yaratilgan **original yengil neyron tarmoq arxitekturasi**. BERT kabi katta modellar 110 million parametr talab qiladi va mobil qurilmalarda ishlay olmaydi. Semurg' Delta V2 esa atigi **6.67 million parametr** bilan raqobatbardosh natija ko'rsatadi.

**Hal qilingan muammo:** O'zbek tilida sifatli, yengil AI modeli yo'q edi. BERT o'zbek morfologiyasini tushunmaydi. Biz bu muammoni uch innovatsiya bilan hal qildik:

- **Delta Gate** — har bir so'z xususiyatini softplus gating orqali filtrlaydi
- **Morfema Tokenizer** — o'zbek so'zlarini ildiz va qo'shimchalarga ajratadi
- **Feature Density Pooling** — muhim tokenlarga ko'proq og'irlik beradi

---

## Benchmark Natijalari

Kaggle GPU T4 x2 | 50,000 Uzum Market sharhi | 5 epoch

| Model | Aniqlik | Parametrlar |
|-------|---------|-------------|
| Baseline (BERT tokenizer) | 90.61% | 30.6M |
| Semurg' Delta V1 (Morfema) | 90.95% | 6.67M |
| **Semurg' Delta V2 (Morfema + Density)** | **91.86%** | **6.67M** |
| Eski rekord (8M, BERT tok) | 89.89% | 8M |
| BERT-base | 91.09% | 110M |

**Asosiy xulosalar:**
- V2 BERT dan **16x yengilroq**, lekin **+0.77% yuqori** aniqlik
- Feature Density Pooling: V1 dan **+0.91%**
- Eski rekorddan: **+1.97%**

---

## Arxitektura

### 1. Delta Gate (V1 asosi)

```python
delta = softplus(W * x)   # softplus = log(1 + exp(x))
gate  = exp(-delta)        # [0, 1] oraliqda
x_out = x * gate           # keraksiz xususiyatlar filtrlandi
```

Identity inicializatsiya barqarorlikni ta'minlaydi — trening boshida model ma'lumotni o'zgarishsiz o'tkazadi.

### 2. Feature Density Pooling (V2 yangiligi)

```python
d   = gate.mean(dim=-1, keepdim=True)  # Feature Density
w   = d * mask                          # Og'irli maska
out = (x * w).sum(1) / w.sum(1)        # Og'irli o'rtacha
```

Gate zichligi yuqori tokenlar — muhimroq ma'lumot tashiydi. Bu oddiy o'rtacha pooling dan **+0.77%** yuqori natija berdi.

### 3. Morfema Tokenizer

```python
'kitobdan'       -> ['kitob', 'dan']
"o'qituvchimiz"  -> ["o'qituvchi", 'imiz']
```

BERT o'zbek qo'shimchalarini taniy olmaydi. Morfema tokenizer ildiz va qo'shimchalarni to'g'ri ajratib, model o'zbek grammatikasini yaxshiroq tushunishiga yordam beradi.

---

## Model Kodi

```python
class SemurgDeltaV2(nn.Module):
    def __init__(self, vocab_size, dim=256, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.W   = nn.Linear(dim, dim)
        self.cls = nn.Linear(dim, num_classes)
        nn.init.eye_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, ids, mask):
        x = self.emb(ids)
        # Delta gate
        g = torch.exp(-torch.log(1 + torch.exp(self.W(x))))
        # Feature Density pooling
        d = g.mean(dim=-1, keepdim=True)
        x = x * g
        m = mask.unsqueeze(-1).float()
        w = d * m
        return self.cls((x * w).sum(1) / w.sum(1).clamp(min=1))
```

---

## Mamba bilan taqqoslash

Mamba (Gu & Dao, 2023) dan ilhom olindi, lekin tubdan farqlanadi:

| Xususiyat | Mamba | Semurg' Delta |
|-----------|-------|---------------|
| Delta maqsadi | Vaqt qadamini boshqaradi | Xususiyat muhimligini aniqlaydi |
| Rekurrent holat | Bor | Yo'q |
| Murakkablik | O(n) rekurrent | O(n) holatsiz |
| Maqsad vazifa | Vaqt seriyalari | Matn klassifikatsiyasi |

---

## Kelajak Rejalari

| Bosqich | Maqsad |
|---------|--------|
| Hozir (V2) | Sentiment klassifikatsiya — 91.86% |
| V3 | Ko'p sinf: mavzu, til darajasi |
| V4 | To'liq encoder — NER, POS tagging |
| V5 | Encoder-Decoder — o'zbek NLP to'plami |
| Keyingi | O'zbek AI foundation modeli |

**Strategik maqsad:** O'zbek tilidagi birinchi original neyron tarmoq arxitekturasi — mobil qurilmalarda ishlaydigan, O'zbekiston IT ekotizimida BERT o'rnini bosadigan model.

---

## Dataset

- **Manba:** `risqaliyevds/uzbek-sentiment-analysis` (Hugging Face)
- **Hajm:** 50,000 Uzum Market mahsulot sharhlari
- **Vazifa:** Ikkilik sentiment klassifikatsiya (musbat / manfiy)

---

## Manba

Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752

---

## Litsenziya

MIT License — [LICENSE](LICENSE) ga qarang
