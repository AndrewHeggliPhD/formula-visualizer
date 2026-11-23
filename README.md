# FormelVis / FormulaViz 📐

Interaktiv webapplikasjon (Streamlit) for visuell utforskning av statistiske sannsynlighetsfordelinger og egne matematiske formler – med **dynamisk visuell skalering** av hver enkelt del av formelen og responsen.

---

## Funksjoner / Features

- **10 innebygde sannsynlighetsfordelinger:** Normal, Gamma, Binomisk, Poisson, Eksponential, Beta, Student's $t$, Kjikvadrat ($\chi^2$), Lognormal og Uniform.
- **Egne formler (fritekst):** Skriv inn egne formler som `tfidf(t, d) = tf(t, d) * log(N/df(t))`, `b = 1/a`, `P(A|B) = P(B|A) * P(A) / P(B)` eller `f(x) = a * x^2 + b * x + c`.
- **Automatisk gjenkjenning av output-type:** Skiller automatisk mellom kontinuerlige fordelinger/kurver, diskrete søylediagrammer og skalar-/konstantformler (med både dynamisk tallverdi og interaktivt 2D-sveip over valgfri variabel).
- **Dynamisk visuell formelskalering (per forekomst!):**
  - *Effekt / Sensitivitet (standard):* Hver forekomst av en parameter i syntakstreet differensieres separat (f.eks. de to $\sigma$-ene i normalfordelingen!). Tekststørrelsen viser relativ effekt (elastisitet), mens fargekoding viser retning (grønn = øker respons, rød = senker respons).
  - *Verdibasert modus:* Størrelsen følger selve tallverdien til hvert ledd og responsen.
- **Fokuspunkt $x_0$ / $k_0$:** Dra fokuspunktet langs kurven for å se hvordan effekten skifter (f.eks. ved $\sigma = |x_0 - \mu|$ i normalfordelingen).
- **Referansekurve («Ghost»-kurve):** Frys en kurve som stiplet referanse for direkte sammenligning.
- **PDF/PMF & CDF + nøkkeltall:** Bytte mellom tetthet og kumulativ fordeling med forventningsverdi, varians og standardavvik.
- **Eksempelbibliotek, lagring, JSON-eksport/import og delbare URL-lenker.**
- **Tospråklig:** Bytt mellom norsk og engelsk direkte i appen.

---

## Publisering på nett via Streamlit Community Cloud

1. Gå til [share.streamlit.io](https://share.streamlit.io) og logg inn med GitHub-kontoen din.
2. Klikk **Create app** → **Yup, I have an app**.
3. Velg dette private repoet, branch `main`, og hovedfil `app.py`.
4. Klikk **Deploy!** (Streamlit installerer automatisk `requirements.txt` og publiserer appen på nett i løpet av 1–2 minutter.)

*(Merk: Gratisplanen til Streamlit Community Cloud inkluderer 1 privat GitHub-app.)*

---

## Kjøre lokalt / Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Kjøre tester / Run Tests

```bash
pip install -r requirements.txt pytest
pytest -v
```
