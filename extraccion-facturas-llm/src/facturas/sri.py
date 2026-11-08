"""
Reglas públicas del SRI (Ecuador) para identificar comprobantes.

* Cédula (10 dígitos): provincia 01–24 o 30, tercer dígito < 6 y dígito verificador módulo 10.
* RUC (13 dígitos):
    - persona natural: cédula válida + establecimiento (p. ej. 001)
    - sociedad privada: tercer dígito 9, módulo 11 con coeficientes 4 3 2 7 6 5 4 3 2 → dígito 10
    - entidad pública:  tercer dígito 6, módulo 11 con coeficientes 3 2 7 6 5 4 3 2 → dígito 9
* Clave de acceso (49 dígitos) de la factura electrónica:
    fecha ddmmaaaa (8) · tipo de comprobante 01 (2) · RUC (13) · ambiente (1) · serie estab+pto (6) ·
    secuencial (9) · código numérico (8) · tipo de emisión (1) · dígito verificador módulo 11 (1)
  En la factura electrónica el número de autorización ES la clave de acceso.
"""
from __future__ import annotations

import random
import re
from datetime import date

CONSUMIDOR_FINAL = "9999999999999"


def _solo_digitos(s: str | None) -> str:
    return re.sub(r"\D", "", s or "")


def digito_modulo10(nueve: str) -> int:
    suma = 0
    for i, c in enumerate(nueve):
        v = int(c) * (2 if i % 2 == 0 else 1)
        suma += v - 9 if v > 9 else v
    return (10 - suma % 10) % 10


def _modulo11(digitos: str, coeficientes: list[int]) -> int:
    resto = sum(int(d) * c for d, c in zip(digitos, coeficientes)) % 11
    v = 11 - resto
    return 0 if v == 11 else v       # 10 → inválido para RUC (se descarta al generar)


def cedula_valida(ced: str | None) -> bool:
    ced = _solo_digitos(ced)
    if len(ced) != 10 or not (1 <= int(ced[:2]) <= 24 or ced[:2] == "30") or int(ced[2]) >= 6:
        return False
    return digito_modulo10(ced[:9]) == int(ced[9])


def tipo_ruc(ruc: str | None) -> str | None:
    """'natural' | 'privada' | 'publica' si el RUC es válido; None si no."""
    ruc = _solo_digitos(ruc)
    if len(ruc) != 13 or not (1 <= int(ruc[:2]) <= 24 or ruc[:2] == "30"):
        return None
    tercero = int(ruc[2])
    if tercero < 6:
        return "natural" if cedula_valida(ruc[:10]) and ruc[10:] != "000" else None
    if tercero == 9:
        v = _modulo11(ruc[:9], [4, 3, 2, 7, 6, 5, 4, 3, 2])
        return "privada" if v != 10 and v == int(ruc[9]) and ruc[10:] != "000" else None
    if tercero == 6:
        v = _modulo11(ruc[:8], [3, 2, 7, 6, 5, 4, 3, 2])
        return "publica" if v != 10 and v == int(ruc[8]) and ruc[9:] != "0000" else None
    return None


def ruc_valido(ruc: str | None) -> bool:
    return tipo_ruc(ruc) is not None


def identificacion_valida(x: str | None) -> bool:
    """Cliente: cédula, RUC o consumidor final (9999999999999)."""
    d = _solo_digitos(x)
    return d == CONSUMIDOR_FINAL or cedula_valida(d) or ruc_valido(d)


# ---------------------------------------------------------------- clave de acceso
def digito_clave(cuarenta_y_ocho: str) -> int:
    """Módulo 11 con pesos 2..7 de derecha a izquierda (ficha técnica de comprobantes electrónicos)."""
    suma, peso = 0, 2
    for c in reversed(cuarenta_y_ocho):
        suma += int(c) * peso
        peso = 2 if peso == 7 else peso + 1
    v = 11 - suma % 11
    return {11: 0, 10: 1}.get(v, v)


def armar_clave(fecha: date, ruc: str, numero_factura: str, codigo: str, ambiente: str = "2") -> str:
    estab, pto, sec = numero_factura.split("-")
    base = f"{fecha:%d%m%Y}01{ruc}{ambiente}{estab}{pto}{sec}{codigo}1"
    return base + str(digito_clave(base))


def desarmar_clave(clave: str | None) -> dict | None:
    """Partes de la clave si tiene 49 dígitos (sin verificar)."""
    c = _solo_digitos(clave)
    if len(c) != 49:
        return None
    return {"fecha": f"{c[4:8]}-{c[2:4]}-{c[0:2]}", "tipo": c[8:10], "ruc": c[10:23], "ambiente": c[23],
            "numero_factura": f"{c[24:27]}-{c[27:30]}-{c[30:39]}", "codigo": c[39:47], "emision": c[47],
            "verificador": c[48]}


def clave_valida(clave: str | None) -> bool:
    c = _solo_digitos(clave)
    return len(c) == 49 and digito_clave(c[:48]) == int(c[48])


# ---------------------------------------------------------------- generación (solo datos sintéticos)
def _provincia(rng: random.Random) -> str:
    return f"{rng.randint(1, 24):02d}"


def generar_cedula(rng: random.Random) -> str:
    nueve = _provincia(rng) + str(rng.randint(0, 5)) + "".join(str(rng.randint(0, 9)) for _ in range(6))
    return nueve + str(digito_modulo10(nueve))


def generar_ruc(rng: random.Random, tipo: str = "privada") -> str:
    while True:
        if tipo == "natural":
            return generar_cedula(rng) + "001"
        if tipo == "privada":
            nueve = _provincia(rng) + "9" + "".join(str(rng.randint(0, 9)) for _ in range(6))
            v = _modulo11(nueve, [4, 3, 2, 7, 6, 5, 4, 3, 2])
            if v != 10:
                return nueve + str(v) + "001"
        else:
            ocho = _provincia(rng) + "6" + "".join(str(rng.randint(0, 9)) for _ in range(5))
            v = _modulo11(ocho, [3, 2, 7, 6, 5, 4, 3, 2])
            if v != 10:
                return ocho + str(v) + "0001"


def tasa_iva(fecha: date) -> float:
    """IVA general del Ecuador: 12 % hasta el 31-III-2024, 15 % desde el 1-IV-2024."""
    return 0.15 if fecha >= date(2024, 4, 1) else 0.12
