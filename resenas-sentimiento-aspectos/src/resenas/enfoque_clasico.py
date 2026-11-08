"""
Enfoque 1 · clásico, sin redes neuronales:

* **Aspectos**: diccionario de expresiones regulares por aspecto aplicado a cada cláusula (escrito mirando solo el
  conjunto dev).
* **Sentimiento**: TF-IDF (palabras 1–2 y caracteres 2–5) + regresión logística entrenada con las estrellas de las
  reseñas no etiquetadas (1–2★ negativo, 3★ neutro, 4–5★ positivo). Es supervisión débil: no hace falta etiquetar.
* **Polaridad del aspecto** = el clasificador aplicado a las cláusulas donde aparece el aspecto; si la cláusula no
  tiene carga clara (probabilidad máxima < umbral), se usa el sentimiento global de la reseña.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import config
from .texto import clausulas, texto_completo

# Patrones por aspecto (inglés, el idioma de las reseñas). \b evita "price" dentro de "priceless", etc.
PATRONES: dict[str, list[str]] = {
    "bateria": [r"batter(y|ies)", r"\bmah\b", r"battery life", r"\bdrains?\b", r"\bjuice\b", r"lasts? (all|a full) day",
                r"hold(s|ing)? (a |its )?charge", r"\bcharges? (my phone )?(\d+|several|multiple) times",
                r"\b\d+(\.\d+)? times\b", r"\bdies\b", r"\bbatt\b", r"\bbateria\b"],
    "carga": [r"\bcharg(e|es|ed|ing|er|ers)\b", r"\bcables?\b", r"\bcords?\b", r"\busb\b", r"lightning", r"\bports?\b",
              r"wireless", r"magsafe", r"\bwatts?\b", r"\b\d+ ?w\b", r"fast charg", r"\bcargador", r"\bcarga\b"],
    "pantalla": [r"\bdisplay", r"bubbles?", r"touch ?(screen|sensitiv)", r"\bresponsive", r"fingerprints?\b(?! (reader|sensor))",
                 r"\bclarity\b", r"\bbright", r"\bdust\b", r"smudge", r"privacy", r"anti-?(glare|spy)", r"\bglare\b",
                 r"screen (is|looks|quality|colou?rs?|resolution)", r"(see|seen) (the )?(screen|items)", r"\bfog",
                 r"\bpantalla\b", r"\bdull\b"],
    "sonido": [r"\bsound", r"\baudio\b", r"\bvolume\b", r"\bbass\b", r"\bmic(rophone)?\b", r"\bcalls?\b", r"\bnoise",
               r"\bhear\b", r"\bloud", r"\bmusic\b", r"\bspeakers?\b", r"\bllamadas\b"],
    "calidad": [r"\bquality\b", r"\bdurab", r"\bsturdy\b", r"\bflims", r"\bcheap(ly)? (made|plastic|material|quality)",
                r"\bbroke\b", r"\bbreaks?\b", r"\bbroken\b", r"stopped working", r"quit (working|charging)", r"\bdefect",
                r"\bfell apart\b", r"\bpeel", r"\bcrack(ed|s)?\b", r"\bsolid\b", r"well made", r"poorly made",
                r"\bmaterial", r"\bwear(ing|s)? (out|off)", r"\bfaulty\b", r"\bflimsy\b", r"\btore\b", r"\bripped\b",
                r"\bheld up\b", r"\bholds? up\b", r"\bholding up\b", r"\bdied\b", r"doesn'?t work", r"didn'?t work",
                r"not working", r"\blast(ed|s)? (only|about|for|just|a)\b", r"\bonly lasted\b", r"no longer work",
                r"\bfragile\b", r"\bfr[aá]gil", r"\byellow", r"\bdiscolou?r", r"\bstain", r"\bfad(ing|ed|es)\b",
                r"\bscratch(es|ed)? (easily|easy)", r"\bscratched\b", r"\bchip", r"\boverheat", r"\b(gets?|got|runs?) (really |very |way to |too )?hot",
                r"\bsmoke\b", r"fire hazard", r"\bjunk\b", r"\bdud\b", r"\bstill going\b", r"\bcalidad\b",
                r"\bsuction (fails|cup is bad|power was weak)", r"adhesive (doesn'?t|melt)", r"\b(does not|doesn'?t|did not|didn'?t) stick"],
    "ajuste": [r"\bfits?\b", r"\bfitting\b", r"\bfitted\b", r"\bcompatib", r"\bbuttons?\b", r"cut ?outs?",
               r"\bsnug", r"\bloose\b", r"\btight\b", r"\bslips?\b", r"falls? (off|out|down)", r"fell off",
               r"doesn'?t cover", r"(not|doesn'?t) (completely |even )?cover", r"camera (hole|opening|cut)",
               r"hold(s|ing)? (my |the )?phone", r"\bstay(s|ed)? (in place|put|on|in)", r"won'?t stay", r"\bwork(s|ed)? (great |well )?with (my|the|any)",
               r"too (big|small|large|short|long|thick)", r"\bwrist\b", r"hurts", r"comfortable (to wear|on)",
               r"\bencaj", r"\bvents?\b", r"\bsecure"],
    "diseno": [r"\blooks? (great|good|nice|amazing|beautiful|exactly|horrible|cheap)", r"\bcolou?rs?\b", r"\bpretty\b",
               r"\bbeautiful\b", r"\bcute\b", r"\bstylish\b", r"\bsleek\b", r"\bslim\b", r"\bbulky\b", r"\bgrippy\b",
               r"\bdesign", r"\bfeels? (nice|great|good|comfortable)", r"\bslippery\b", r"\bgorgeous\b", r"\bstyle\b",
               r"\blight ?weight\b", r"\bheavy\b", r"\bcompact\b", r"\btravel\b", r"\bclassy\b", r"\belegant\b",
               r"\bsmall size\b", r"\btama[nñ]o\b"],
    "proteccion": [r"\bprotect", r"\bdropp?(ed|s|ing)?\b", r"\bsurviv", r"\bshock", r"saved my", r"\bwithstand",
                   r"\bproteg"],
    "precio": [r"\bprice", r"\bcost", r"\bmoney\b", r"\bcheap(er)?\b(?! (made|plastic|material|quality))",
               r"\bexpensive\b", r"\bvalue\b", r"\bworth\b", r"\bbargain\b", r"\bdeal\b", r"\$\d+", r"\bpaid\b",
               r"\bafford", r"\bprecio\b", r"\binversi[oó]n\b", r"\bbucks\b"],
    "envio": [r"\bship(ping|ped)\b", r"\bdeliver", r"\barriv", r"\bpackag", r"\bcame (in|with|broken|damaged|scratched)",
              r"\bmissing\b", r"\bonly (received|came|got)\b", r"(was|looked|seemed|been|must be) used\b", r"refurbished",
              r"brand new", r"like new", r"out of the box", r"\bwrong (colou?r|item|model|size)", r"(not the|different) colou?r",
              r"\bsealed\b", r"\bpack\b", r"\blleg[oó]", r"\bcaja\b", r"(didn'?t|did not|doesn'?t) come with", r"no (viene|ven[ií]a)"],
    "atencion": [r"customer service", r"costumer service", r"\bseller\b", r"\bwarranty\b", r"\brefund", r"\bsupport\b",
                 r"\bcontacted\b", r"\bmanufacturer\b", r"(sent|send) (me )?(a )?(new|replacement)", r"return (window|policy|date)",
                 r"\breplace(d|ment)? (it|the|my)? ?(within|quickly|promptly|for free)", r"\bvendedor\b", r"\bservicio\b",
                 r"(stand|stood) by (their|its)", r"\bresponded\b", r"\bcompany (sent|replaced|responded)"],
    "facilidad_uso": [r"\binstall", r"\bapply", r"\bapplication\b", r"\bsetup\b", r"set up", r"\bpair(ed|ing|s)?\b",
                      r"\binstructions?\b", r"easy to (use|put|apply|install|take|change|attach|handle|hold)",
                      r"\bhard to (put|get|take|apply|use|install)", r"\bconvenient\b", r"\bconnect(s|ed|ion|ing|ivity)?\b",
                      r"\balign", r"hands? free", r"cutting (out|on and off)", r"\bdisconnect", r"\bput (it )?on\b",
                      r"\beasy\b", r"\bsimple\b", r"\bassembl"],
}
_PATRONES_C = {a: re.compile("|".join(p), re.I) for a, p in PATRONES.items()}


def aspectos_en(clausula: str) -> list[str]:
    return [a for a, rx in _PATRONES_C.items() if rx.search(clausula)]


def etiqueta_estrellas(estrellas: int) -> str:
    return "negativo" if estrellas <= 2 else "neutro" if estrellas == 3 else "positivo"


@dataclass
class ModeloClasico:
    umbral_clausula: float = 0.55
    vec_pal: TfidfVectorizer = field(default_factory=lambda: TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True, strip_accents="unicode"))
    vec_car: TfidfVectorizer = field(default_factory=lambda: TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=200_000, sublinear_tf=True))
    clf: LogisticRegression = field(default_factory=lambda: LogisticRegression(
        C=4.0, max_iter=2000, class_weight="balanced"))

    def _x(self, textos, ajustar=False):
        if ajustar:
            return hstack([self.vec_pal.fit_transform(textos), self.vec_car.fit_transform(textos)]).tocsr()
        return hstack([self.vec_pal.transform(textos), self.vec_car.transform(textos)]).tocsr()

    def entrenar(self, df: pd.DataFrame) -> "ModeloClasico":
        """df con titulo_resena, texto, estrellas (reseñas NO etiquetadas a mano)."""
        textos = [texto_completo(t, x) for t, x in zip(df["titulo_resena"], df["texto"])]
        y = df["estrellas"].map(etiqueta_estrellas)
        self.clf.fit(self._x(textos, ajustar=True), y)
        return self

    def probabilidades(self, textos: list[str]) -> np.ndarray:
        return self.clf.predict_proba(self._x(textos))

    @property
    def clases(self) -> list[str]:
        return list(self.clf.classes_)

    def analizar(self, titulo: str, texto: str) -> dict:
        """{"sentimiento_global": ..., "aspectos": [{aspecto, polaridad, cita}]} como el LLM."""
        return self.analizar_lote([(titulo, texto)])[0]

    def analizar_lote(self, pares: list[tuple[str, str]]) -> list[dict]:
        completos = [texto_completo(t, x) for t, x in pares]
        p_global = self.probabilidades(completos)
        por_resena = [clausulas(c) for c in completos]
        todas = [c for cs in por_resena for c in cs]
        p_cl = self.probabilidades(todas) if todas else np.zeros((0, len(self.clases)))
        salida, k = [], 0
        for i, cs in enumerate(por_resena):
            glob = self.clases[int(np.argmax(p_global[i]))]
            menciones: dict[str, dict] = {}
            puntaje: dict[str, np.ndarray] = {}
            for c in cs:
                p = p_cl[k]
                k += 1
                for a in aspectos_en(c):
                    puntaje[a] = puntaje.get(a, 0) + p
                    menciones.setdefault(a, {"aspecto": a, "cita": c})
            for a, m in menciones.items():
                p = puntaje[a] / puntaje[a].sum()
                m["polaridad"] = self.clases[int(np.argmax(p))] if p.max() >= self.umbral_clausula else glob
            salida.append({"sentimiento_global": glob, "aspectos": list(menciones.values())})
        return salida

    def guardar(self, ruta) -> None:
        joblib.dump(self, ruta)

    @staticmethod
    def cargar(ruta) -> "ModeloClasico":
        return joblib.load(ruta)


def cronometrar(modelo: ModeloClasico, pares: list[tuple[str, str]]) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    r = modelo.analizar_lote(pares)
    return r, (time.perf_counter() - t0) / max(1, len(pares))


RUTA_MODELO = config.DIR_DATOS / "modelo_clasico.joblib"   # versionado (8 MB): la app funciona sin descargar
