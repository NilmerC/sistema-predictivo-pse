import json
import numpy as np
import re
import scipy.stats as stats
from typing import Any, Dict, Tuple

# =====================================================================
# ⚙️ CONFIGURACIÓN Y PROMEDIOS ESTÁNDAR (LIGA 1 PERÚ)
# =====================================================================
PROMEDIOS_ESTANDAR = {
    "altura_msnm": 0,
    "es_partido_abierto": False,
    "goles_favor_local": 1.40,
    "goles_contra_local": 1.00,
    "goles_favor_visita": 1.00,
    "goles_contra_visita": 1.40,
    "goles_promedio_liga": 1.28,  # Media de goles por equipo (~2.56 partido)
    "corners_favor_local": 5.50,
    "corners_contra_local": 4.20,
    "corners_favor_visita": 4.00,
    "corners_contra_visita": 5.80,
    "remates_favor_local": 12.00,
    "remates_favor_visita": 9.00,
    "tiros_arco_local": 4.50,
    "tiros_arco_visita": 3.00,
    "tarjetas_promedio_arbitro": 5.50,  # Promedio real de árbitros Liga 1
    "tarjetas_recibidas_local": 2.30,
    "tarjetas_provocadas_local": 2.30,
    "tarjetas_recibidas_visita": 2.70,
    "tarjetas_provocadas_visita": 2.10,
    "elo_local": 1400,
    "elo_visita": 1400,
}

DATOS_PARTIDO = {
    # --- Datos Básicos ---
    "liga": "UEFA Champions / Europa League Qualifiers",
    "equipo_local": "Fenerbahce",
    "equipo_visitante": "Lyon",

    # --- Configuración Geográfica / Condición del Partido ---
    "altura_msnm": 30, # Altitud de Estambul en metros sobre el nivel del mar
    "es_partido_abierto": True, # Encuentro internacional de alta intensidad y vocación ofensiva

    # --- Goles Promedio (Últimos 10 partidos) ---
    "goles_favor_local": 2.05,   # Goles anotados por el local jugando en su estadio
    "goles_contra_local": 0.85,  # Goles recibidos por el local jugando en su estadio
    "goles_favor_visita": 1.40,  # Goles anotados por el visitante fuera de casa
    "goles_contra_visita": 1.30, # Goles recibidos por el visitante fuera de casa
    "goles_promedio_liga": 1.45, # Promedio general de goles por equipo por partido en la competición

    # --- Saques de Esquina (Córners) ---
    "corners_favor_local": 6.4,   # Córners a favor del local en casa
    "corners_contra_local": 3.8,  # Córners en contra del local en casa
    "corners_favor_visita": 4.9,  # Córners a favor del visitante fuera
    "corners_contra_visita": 5.2, # Córners en contra del visitante fuera

    # --- Volumen de Remates y Tiros (Totales por partido) ---
    "remates_favor_local": 16.2,  # Disparos totales a favor del local
    "remates_favor_visita": 13.5, # Disparos totales a favor del visitante
    "tiros_arco_local": 6.0,      # Disparos DIRECTOS a puerta del local
    "tiros_arco_visita": 4.7,     # Disparos DIRECTOS a puerta del visitante

    # --- Disciplina (Tarjetas Amarillas + Rojas) ---
    "tarjetas_promedio_arbitro": 5.6, # Promedio histórico de tarjetas por partido en UEFA
    "tarjetas_recibidas_local": 2.8,
    "tarjetas_provocadas_local": 2.9,
    "tarjetas_recibidas_visita": 2.6,
    "tarjetas_provocadas_visita": 2.7,

    # --- Fuerza de Equipos ---
    "elo_local": 1695, # Rating ELO aproximado del local
    "elo_visita": 1680  # Rating ELO aproximado del visitante
}
# =====================================================================
# MOTOR MATH & ESTADÍSTICA (PSE ENGINE 6.2 - LIGA 1 EDITION)
# =====================================================================


def safe_division(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
  if denominator is None or denominator <= 1e-9 or np.isnan(denominator):
    return default
  return numerator / denominator


def nbinom_pmf(k: int, mu: float, alpha: float) -> float:
  if mu <= 0:
    return 1.0 if k == 0 else 0.0
  r = 1.0 / alpha
  p = r / (r + mu)
  return stats.nbinom.pmf(k, r, p)


def nbinom_cdf(k: int, mu: float, alpha: float) -> float:
  if mu <= 0:
    return 1.0
  r = 1.0 / alpha
  p = r / (r + mu)
  return stats.nbinom.cdf(k, r, p)


def obtener_marcador_skellam(lambda_local: float, lambda_visita: float) -> str:
  """Calcula la diferencia de goles más probable usando la Distribución de Skellam."""
  diferencias = range(-5, 6)
  probs = [stats.skellam.pmf(k, lambda_local, lambda_visita) for k in diferencias]
  diff_optima = diferencias[int(np.argmax(probs))]

  if diff_optima > 0:
    return (
        f"{diff_optima} - 0"
        if diff_optima == 1
        else f"{diff_optima} - {diff_optima - 1}"
    )
  elif diff_optima < 0:
    return (
        f"0 - {abs(diff_optima)}"
        if abs(diff_optima) == 1
        else f"{abs(diff_optima) - 1} - {abs(diff_optima)}"
    )
  else:
    return "1 - 1"


def calcular_matriz_sobredispersa(
    lambda_local: float,
    lambda_visita: float,
    rho: float = -0.11,
    alpha: float = 0.15,
) -> Tuple[float, float, float, str, float, str, float]:
  """Matriz Bivariada con corrección adaptativa Dixon-Coles y Binomial Negativa."""
  max_goles = 14
  matriz = np.zeros((max_goles, max_goles))
  matriz_poisson = np.zeros((max_goles, max_goles))

  if (lambda_local + lambda_visita) >= 3.0:
    rho = 0.0

  for i in range(max_goles):
    for j in range(max_goles):
      p_i = nbinom_pmf(i, lambda_local, alpha)
      p_j = nbinom_pmf(j, lambda_visita, alpha)

      matriz_poisson[i, j] = stats.poisson.pmf(
          i, lambda_local
      ) * stats.poisson.pmf(j, lambda_visita)

      tau = 1.0
      if rho != 0.0:
        if i == 0 and j == 0:
          tau = 1.0 - (lambda_local * lambda_visita * rho)
        elif i == 1 and j == 0:
          tau = 1.0 + (lambda_visita * rho)
        elif i == 0 and j == 1:
          tau = 1.0 + (lambda_local * rho)
        elif i == 1 and j == 1:
          tau = 1.0 - rho

      matriz[i, j] = max(p_i * p_j * tau, 0.0)

  suma = np.sum(matriz)
  if suma > 0:
    matriz /= suma

  prob_local = np.sum(np.tril(matriz, -1))
  prob_empate = np.sum(np.diag(matriz))
  prob_visita = np.sum(np.triu(matriz, 1))

  idx = np.unravel_index(np.argmax(matriz), matriz.shape)
  marcador_probable = f"{idx[0]} - {idx[1]}"
  prob_marcador = float(matriz[idx])

  idx_p = np.unravel_index(np.argmax(matriz_poisson), matriz_poisson.shape)
  marcador_poisson = f"{idx_p[0]} - {idx_p[1]}"
  prob_marcador_poisson = float(matriz_poisson[idx_p])

  return (
      prob_local,
      prob_empate,
      prob_visita,
      marcador_probable,
      prob_marcador,
      marcador_poisson,
      prob_marcador_poisson,
  )


def ejecutar_sistema_pse():
  d = {}
  for clave, valor in DATOS_PARTIDO.items():
    if (
        valor is None
        or (isinstance(valor, (int, float)) and np.isnan(valor))
    ) and clave in PROMEDIOS_ESTANDAR:
      d[clave] = PROMEDIOS_ESTANDAR[clave]
    else:
      d[clave] = valor

  for clave, valor_def in PROMEDIOS_ESTANDAR.items():
    if clave not in d or d[clave] is None:
      d[clave] = valor_def

  altura = d.get("altura_msnm", 0)
  factor_altitud_local = 1.0 + min(
      max(altura - 1000, 0) / 10000.0 * 0.65, 0.20
  )
  factor_altitud_desgaste_visita = 1.0 + min(
      max(altura - 1000, 0) / 10000.0 * 0.75, 0.20
  )

  diff_elo = d.get("elo_local", 1400) - d.get("elo_visita", 1400)
  factor_elo_local = 1.0 + (diff_elo / 2000.0)
  factor_elo_visita = 1.0 - (diff_elo / 2000.0)

  prom_liga = max(d["goles_promedio_liga"], 0.1)

  lambda_goles_l = (
      (d["goles_favor_local"] * d["goles_contra_visita"]) / prom_liga
  ) * factor_altitud_local * factor_elo_local
  lambda_goles_v = (
      (d["goles_favor_visita"] * d["goles_contra_local"]) / prom_liga
  ) * factor_altitud_desgaste_visita * factor_elo_visita

  if d.get("es_partido_abierto", False):
    lambda_goles_l *= 1.08
    lambda_goles_v *= 1.08

  lambda_goles_totales = lambda_goles_l + lambda_goles_v

  lambda_corners_l = (
      (d["corners_favor_local"] + d["corners_contra_visita"]) / 2.0
  ) * (factor_altitud_local**0.5)
  lambda_corners_v = (
      d["corners_favor_visita"] + d["corners_contra_local"]
  ) / 2.0
  lambda_corners_totales = lambda_corners_l + lambda_corners_v

  factor_arbitro = d["tarjetas_promedio_arbitro"] / 5.50
  lambda_tarjetas_l = (
      (d["tarjetas_recibidas_local"] + d["tarjetas_provocadas_visita"]) / 2.0
  ) * factor_arbitro
  lambda_tarjetas_v = (
      (d["tarjetas_recibidas_visita"] + d["tarjetas_provocadas_local"]) / 2.0
  ) * factor_arbitro
  lambda_tarjetas_totales = lambda_tarjetas_l + lambda_tarjetas_v

  lambda_remates_l = d["remates_favor_local"] * (factor_altitud_local**0.3)
  lambda_remates_v = d["remates_favor_visita"]
  lambda_remates_totales = lambda_remates_l + lambda_remates_v

  lambda_tiros_l = d["tiros_arco_local"] * (factor_altitud_local**0.4)
  lambda_tiros_v = d["tiros_arco_visita"]
  lambda_tiros_totales = lambda_tiros_l + lambda_tiros_v

  alpha_goles = 0.15
  alpha_volumen = 0.10

  prob_g_local_mas_0_5 = 1.0 - nbinom_cdf(0, lambda_goles_l, alpha_goles)
  prob_g_local_mas_1_5 = 1.0 - nbinom_cdf(1, lambda_goles_l, alpha_goles)
  prob_g_visita_mas_0_5 = 1.0 - nbinom_cdf(0, lambda_goles_v, alpha_goles)
  prob_g_visita_mas_1_5 = 1.0 - nbinom_cdf(1, lambda_goles_v, alpha_goles)
  prob_g_totales_mas_1_5 = 1.0 - nbinom_cdf(1, lambda_goles_totales, alpha_goles)
  prob_g_totales_menos_4_5 = nbinom_cdf(4, lambda_goles_totales, alpha_goles)

  prob_corners_totales_mas_7_5 = 1.0 - nbinom_cdf(
      7, lambda_corners_totales, alpha_volumen
  )
  prob_corners_totales_mas_8_5 = 1.0 - nbinom_cdf(
      8, lambda_corners_totales, alpha_volumen
  )
  prob_corners_l_mas_3_5 = 1.0 - nbinom_cdf(3, lambda_corners_l, alpha_volumen)
  prob_corners_v_mas_3_5 = 1.0 - nbinom_cdf(3, lambda_corners_v, alpha_volumen)

  prob_tarjetas_totales_mas_2_5 = 1.0 - nbinom_cdf(
      2, lambda_tarjetas_totales, alpha_volumen
  )
  prob_tarjetas_totales_mas_3_5 = 1.0 - nbinom_cdf(
      3, lambda_tarjetas_totales, alpha_volumen
  )
  prob_tarjetas_l_mas_1_5 = 1.0 - nbinom_cdf(1, lambda_tarjetas_l, alpha_volumen)
  prob_tarjetas_v_mas_1_5 = 1.0 - nbinom_cdf(1, lambda_tarjetas_v, alpha_volumen)

  prob_remates_totales_mas_19_5 = 1.0 - nbinom_cdf(
      19, lambda_remates_totales, alpha_volumen
  )
  prob_remates_totales_mas_22_5 = 1.0 - nbinom_cdf(
      22, lambda_remates_totales, alpha_volumen
  )
  prob_remates_l_mas_9_5 = 1.0 - nbinom_cdf(9, lambda_remates_l, alpha_volumen)
  prob_remates_v_mas_10_5 = 1.0 - nbinom_cdf(
      10, lambda_remates_v, alpha_volumen
  )

  prob_tiros_totales_mas_7_5 = 1.0 - nbinom_cdf(
      7, lambda_tiros_totales, alpha_volumen
  )
  prob_tiros_totales_mas_9_5 = 1.0 - nbinom_cdf(
      9, lambda_tiros_totales, alpha_volumen
  )
  prob_tiros_l_mas_2_5 = 1.0 - nbinom_cdf(2, lambda_tiros_l, alpha_volumen)
  prob_tiros_l_mas_3_5 = 1.0 - nbinom_cdf(3, lambda_tiros_l, alpha_volumen)
  prob_tiros_v_mas_2_5 = 1.0 - nbinom_cdf(2, lambda_tiros_v, alpha_volumen)
  prob_tiros_v_mas_3_5 = 1.0 - nbinom_cdf(3, lambda_tiros_v, alpha_volumen)

  (
      prob_poi_l,
      prob_poi_e,
      prob_poi_v,
      marcador_dc,
      prob_marcador,
      marcador_poisson,
      prob_marcador_poisson,
  ) = calcular_matriz_sobredispersa(lambda_goles_l, lambda_goles_v)
  marcador_skellam = obtener_marcador_skellam(lambda_goles_l, lambda_goles_v)

  def q(prob: float) -> str:
    c = safe_division(1.0, prob, default=float("inf"))
    return f"{c:.2f}" if c < 999 else "N/A"

  print("\n" + "=" * 80)
  print(
      f"📋 INFORME DE PROBABILIDADES DE VOLUMEN (SISTEMA PSE -"
      f" {DATOS_PARTIDO['liga']})"
  )
  print("=" * 80)
  print(
      f"Partido: {d['equipo_local']} vs {d['equipo_visitante']} | Liga:"
      f" {d['liga']}"
  )
  print(
      f"Marcador Probable (Dixon-Coles + NB): {marcador_dc}"
      f" ({prob_marcador*100:.1f}%) | Marcadores Clave: Poisson:"
      f" {marcador_poisson} ({prob_marcador_poisson*100:.1f}%) | Skellam:"
      f" {marcador_skellam}"
  )
  print(
      f"Probabilidades 1X2: Local: {prob_poi_l*100:.1f}% | Empate:"
      f" {prob_poi_e*100:.1f}% | Visita: {prob_poi_v*100:.1f}%"
  )
  print("-" * 80)

  print("📊 1. ANÁLISIS DETALLADO DE GOLES PROMEDIO")
  print(f"  * Goles Esperados Local ({d['equipo_local']}): {lambda_goles_l:.2f}")
  print(
      f"  * Goles Esperados Visitante ({d['equipo_visitante']}):"
      f" {lambda_goles_v:.2f}"
  )
  print(f"  * Goles Esperados Totales: {lambda_goles_totales:.2f}")
  print(
      f"  * Prob. {d['equipo_local']} anota más de 0.5 goles:"
      f" {prob_g_local_mas_0_5*100:.1f}%  | Cuota Justa:"
      f" {q(prob_g_local_mas_0_5)}"
  )
  print(
      f"  * Prob. {d['equipo_local']} anota más de 1.5 goles:"
      f" {prob_g_local_mas_1_5*100:.1f}%  | Cuota Justa:"
      f" {q(prob_g_local_mas_1_5)}"
  )
  print(
      f"  * Prob. {d['equipo_visitante']} anota más de 0.5 goles:"
      f" {prob_g_visita_mas_0_5*100:.1f}%  | Cuota Justa:"
      f" {q(prob_g_visita_mas_0_5)}"
  )
  print(
      f"  * Prob. {d['equipo_visitante']} anota más de 1.5 goles:"
      f" {prob_g_visita_mas_1_5*100:.1f}%  | Cuota Justa:"
      f" {q(prob_g_visita_mas_1_5)}"
  )
  print(
      f"  * Prob. Goles Totales MÁS de 1.5: {prob_g_totales_mas_1_5*100:.1f}% "
      f" | Cuota Justa: {q(prob_g_totales_mas_1_5)}"
  )
  print(
      f"  * Prob. Goles Totales MENOS de 4.5:"
      f" {prob_g_totales_menos_4_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_g_totales_menos_4_5)}"
  )
  print("-" * 80)

  print("📐 2. ANÁLISIS DETALLADO DE CÓRNERS")
  print(
      f"  * Proyección Corners {d['equipo_local']}: {lambda_corners_l:.2f} |"
      f" {d['equipo_visitante']}: {lambda_corners_v:.2f} | Total:"
      f" {lambda_corners_totales:.2f}"
  )
  print(
      f"  * Prob. Córners Totales MÁS de 7.5:"
      f" {prob_corners_totales_mas_7_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_corners_totales_mas_7_5)}"
  )
  print(
      f"  * Prob. Córners Totales MÁS de 8.5:"
      f" {prob_corners_totales_mas_8_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_corners_totales_mas_8_5)}"
  )
  print(
      f"  * Prob. {d['equipo_local']} Corners MÁS de 3.5:"
      f" {prob_corners_l_mas_3_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_corners_l_mas_3_5)}"
  )
  print(
      f"  * Prob. {d['equipo_visitante']} Corners MÁS de 3.5:"
      f" {prob_corners_v_mas_3_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_corners_v_mas_3_5)}"
  )
  print("-" * 80)

  print("🟨 3. ANÁLISIS DETALLADO DE TARJETAS")
  print(
      f"  * Proyección Tarjetas {d['equipo_local']}: {lambda_tarjetas_l:.2f} |"
      f" {d['equipo_visitante']}: {lambda_tarjetas_v:.2f} | Total:"
      f" {lambda_tarjetas_totales:.2f}"
  )
  print(
      f"  * Prob. Tarjetas Totales MÁS de 2.5:"
      f" {prob_tarjetas_totales_mas_2_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_tarjetas_totales_mas_2_5)}"
  )
  print(
      f"  * Prob. Tarjetas Totales MÁS de 3.5:"
      f" {prob_tarjetas_totales_mas_3_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_tarjetas_totales_mas_3_5)}"
  )
  print(
      f"  * Prob. {d['equipo_local']} Tarjetas MÁS de 1.5:"
      f" {prob_tarjetas_l_mas_1_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_tarjetas_l_mas_1_5)}"
  )
  print(
      f"  * Prob. {d['equipo_visitante']} Tarjetas MÁS de 1.5:"
      f" {prob_tarjetas_v_mas_1_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_tarjetas_v_mas_1_5)}"
  )
  print("-" * 80)

  print("🏹 4. ANÁLISIS DETALLADO DE REMATES (TOTALES Y POR EQUIPO)")
  print(
      f"  * Promedio Remates {d['equipo_local']}: {lambda_remates_l:.2f} |"
      f" {d['equipo_visitante']}: {lambda_remates_v:.2f} | Total Esperado:"
      f" {lambda_remates_totales:.2f}"
  )
  print(
      f"  * Prob. Remates Totales MÁS de 19.5:"
      f" {prob_remates_totales_mas_19_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_remates_totales_mas_19_5)}"
  )
  print(
      f"  * Prob. Remates Totales MÁS de 22.5:"
      f" {prob_remates_totales_mas_22_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_remates_totales_mas_22_5)}"
  )
  print(
      f"  * Prob. {d['equipo_local']} Remates MÁS de 9.5:"
      f" {prob_remates_l_mas_9_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_remates_l_mas_9_5)}"
  )
  print(
      f"  * Prob. {d['equipo_visitante']} Remates MÁS de 10.5:"
      f" {prob_remates_v_mas_10_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_remates_v_mas_10_5)}"
  )
  print("-" * 80)

  print("🎯 5. ANÁLISIS DETALLADO DE TIROS AL ARCO (TOTALES Y POR EQUIPO)")
  print(
      f"  * Promedio Tiros al Arco {d['equipo_local']}: {lambda_tiros_l:.2f} |"
      f" {d['equipo_visitante']}: {lambda_tiros_v:.2f} | Total Esperado:"
      f" {lambda_tiros_totales:.2f}"
  )
  print(
      f"  * Prob. Tiros al Arco Totales MÁS de 7.5:"
      f" {prob_tiros_totales_mas_7_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_tiros_totales_mas_7_5)}"
  )
  print(
      f"  * Prob. Tiros al Arco Totales MÁS de 9.5:"
      f" {prob_tiros_totales_mas_9_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_tiros_totales_mas_9_5)}"
  )
  print(
      f"  * Prob. {d['equipo_local']} Tiros al Arco MÁS de 2.5:"
      f" {prob_tiros_l_mas_2_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_tiros_l_mas_2_5)}"
  )
  print(
      f"  * Prob. {d['equipo_local']} Tiros al Arco MÁS de 3.5:"
      f" {prob_tiros_l_mas_3_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_tiros_l_mas_3_5)}"
  )
  print(
      f"  * Prob. {d['equipo_visitante']} Tiros al Arco MÁS de 2.5:"
      f" {prob_tiros_v_mas_2_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_tiros_v_mas_2_5)}"
  )
  print(
      f"  * Prob. {d['equipo_visitante']} Tiros al Arco MÁS de 3.5:"
      f" {prob_tiros_v_mas_3_5*100:.1f}% | Cuota Justa:"
      f" {q(prob_tiros_v_mas_3_5)}"
  )
  print("=" * 80 + "\n")

  # =====================================================================
  # AUTOMATIZACIÓN, HISTORIAL E INYECCIÓN DE DATOS EN index.html
  # =====================================================================
  nuevo_match_data = {
      "id": f"{d['equipo_local']}_vs_{d['equipo_visitante']}".replace(" ", "_"),
      "league": d["liga"],
      "badgeLeague": f"SISTEMA PSE - {d['liga'].upper()}",
      "teamHome": d["equipo_local"],
      "teamAway": d["equipo_visitante"],
      "p1": round(float(prob_poi_l * 100), 2),
      "pX": round(float(prob_poi_e * 100), 2),
      "p2": round(float(prob_poi_v * 100), 2),
      "poissonScore": marcador_poisson,  # <--- INSÉRTALO AQUÍ (Línea 515)
      "scoreDixonColes": marcador_dc,       # <--- Usa "marcador_dc" aquí
    "scorePoisson": marcador_poisson,      # Ajusta el nombre de tu variable en Python si es diferente
    "scoreSkellam": marcador_skellam,      # Ajusta el nombre de tu variable en Python si es diferente
      "metrics": {
          "xgHome": round(float(lambda_goles_l), 2),
          "xgAway": round(float(lambda_goles_v), 2),
          "cornersHome": round(float(lambda_corners_l), 2),
          "cornersAway": round(float(lambda_corners_v), 2),
          "cardsHome": round(float(lambda_tarjetas_l), 2),
          "cardsAway": round(float(lambda_tarjetas_v), 2),
          "shotsHome": round(float(lambda_remates_l), 2),
          "shotsAway": round(float(lambda_remates_v), 2),
          "shotsTargetHome": round(float(lambda_tiros_l), 2),
          "shotsTargetAway": round(float(lambda_tiros_v), 2),
      },
      "markets": [
          {
              "id": "g1",
              "category": "goals",
              "label": (
                  f"{d['equipo_local']} anota >0.5"
                  f" goles"
              ),
              "prob": round(float(prob_g_local_mas_0_5 * 100), 1),
              "odds": float(q(prob_g_local_mas_0_5)),
              "team": d["equipo_local"],
          },
          {
              "id": "g2",
              "category": "goals",
              "label": "Goles Totales MÁS de 1.5",
              "prob": round(float(prob_g_totales_mas_1_5 * 100), 1),
              "odds": float(q(prob_g_totales_mas_1_5)),
              "team": "Partido",
          },
          {
              "id": "g3",
              "category": "goals",
              "label": "Goles Totales MENOS de 4.5",
              "prob": round(float(prob_g_totales_menos_4_5 * 100), 1),
              "odds": float(q(prob_g_totales_menos_4_5)),
              "team": "Partido",
          },
          {
              "id": "g4",
              "category": "goals",
              "label": (
                  f"{d['equipo_visitante']} anota >0.5"
                  f" goles"
              ),
              "prob": round(float(prob_g_visita_mas_0_5 * 100), 1),
              "odds": float(q(prob_g_visita_mas_0_5)),
              "team": d["equipo_visitante"],
          },
          {
              "id": "c1",
              "category": "corners",
              "label": "Córners Totales MÁS de 7.5",
              "prob": round(float(prob_corners_totales_mas_7_5 * 100), 1),
              "odds": float(q(prob_corners_totales_mas_7_5)),
              "team": "Partido",
          },
          {
              "id": "c2",
              "category": "corners",
              "label": "Córners Totales MÁS de 8.5",
              "prob": round(float(prob_corners_totales_mas_8_5 * 100), 1),
              "odds": float(q(prob_corners_totales_mas_8_5)),
              "team": "Partido",
          },
          {
              "id": "k1",
              "category": "cards",
              "label": "Tarjetas Totales MÁS de 2.5",
              "prob": round(float(prob_tarjetas_totales_mas_2_5 * 100), 1),
              "odds": float(q(prob_tarjetas_totales_mas_2_5)),
              "team": "Partido",
          },
          {
              "id": "k2",
              "category": "cards",
              "label": "Tarjetas Totales MÁS de 3.5",
              "prob": round(float(prob_tarjetas_totales_mas_3_5 * 100), 1),
              "odds": float(q(prob_tarjetas_totales_mas_3_5)),
              "team": "Partido",
          },
      ],
  }

  # Cargar historial existente o crear uno nuevo
  historial = []
  try:
      with open("historial_partidos.json", "r", encoding="utf-8") as f:
          historial = json.load(f)
  except FileNotFoundError:
      historial = []

  # Evitar duplicados del mismo partido y ponerlo al inicio
  historial = [p for p in historial if p["id"] != nuevo_match_data["id"]]
  historial.insert(0, nuevo_match_data)

  # Guardar historial actualizado en disco
  with open("historial_partidos.json", "w", encoding="utf-8") as f:
      json.dump(historial, f, indent=4, ensure_ascii=False)

  # Inyectar el historial completo y el partido actual en el HTML usando marcadores seguros
  historial_json_str = json.dumps(historial, ensure_ascii=False)
  nuevo_bloque_js = f"var historialPartidos = {json.dumps(historial, ensure_ascii=False)};\nvar currentMatchData = historialPartidos[0] || null;"

  try:
      with open("index.html", "r", encoding="utf-8") as file:
          html_content = file.read()

      inicio_marca = "<!-- PSE_DATA_START -->"
      fin_marca = "<!-- PSE_DATA_END -->"
      
      patron = f"{inicio_marca}.*?{fin_marca}"
      reemplazo = f"{inicio_marca}\n{nuevo_bloque_js}\n{fin_marca}"
      
      if inicio_marca in html_content:
          html_content_actualizado = re.sub(patron, reemplazo, html_content, flags=re.DOTALL)
          with open("index.html", "w", encoding="utf-8") as file:
              file.write(html_content_actualizado)
          print("✅ ¡Partido guardado en el historial, index.html actualizado y sincronizado automáticamente!")
      else:
          print("⚠️ Advertencia: No se encontraron las marcas <!-- PSE_DATA_START --> en tu index.html. Solo se mostró el reporte por consola.")
  except FileNotFoundError:
      print(
          "⚠️ Advertencia: No se encontró el archivo 'index.html' en la misma"
          " ruta. Solo se mostró el reporte por consola."
      )


if __name__ == "__main__":
  ejecutar_sistema_pse()