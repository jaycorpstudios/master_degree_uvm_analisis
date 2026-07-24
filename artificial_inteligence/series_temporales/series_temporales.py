"""Prevision de una serie temporal mensual de pasajeros (Actividad 4).

Toma el archivo PASAJEROS (144 meses), construye los atributos que pide la
actividad (año, mes y un indice de serie), separa entrenamiento y prueba
respetando el orden del tiempo, y ajusta dos modelos con roles distintos:

  * una regresion lineal sobre log(pasajeros) ~ serie + estacionalidad mensual,
    como modelo de aprendizaje automatico que se evalua contra la prueba;
  * un suavizado exponencial Holt-Winters (ETS), que genera la prevision con
    su intervalo de confianza (el mismo enfoque que la "Prevision" de una hoja
    de calculo).

Exporta un CSV con los valores reales y las predicciones de ambos modelos,
guarda las graficas y persiste los modelos entrenados para reusarlos despues.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "PASAJEROS.csv"
EXPORT_PATH = BASE_DIR / "data" / "prevision_pasajeros.csv"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"

ANIO_CORTE = 1960          # < 1960 -> entrenamiento ; >= 1960 -> prueba
HORIZONTE_FUTURO = 12      # meses de prevision mas alla de los datos (1961)
SEASONAL_PERIODS = 12

COLOR_REAL = "#292929"
COLOR_HW = "#C83A37"
COLOR_REG = "#439890"
COLOR_BANDA = "#F09997"
COLOR_GUIA = "#A3A3A3"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_dataset():
    section("PASO 1 - CARGA DE DATOS")
    df = pd.read_csv(DATA_PATH)
    df.columns = ["fecha", "pasajeros"]
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m")
    print(f"Filas leidas: {len(df)}")
    print(f"Rango de fechas: {df['fecha'].min():%Y-%m} a {df['fecha'].max():%Y-%m}")
    print("\nPrimeras filas del archivo original:")
    print(df.head().to_string(index=False))
    return df


def build_columns(df):
    section("PASO 2 - ATRIBUTOS: AÑO, MES Y SERIE")
    df = df.copy()
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["serie"] = np.arange(1, len(df) + 1)
    df = df[["serie", "anio", "mes", "pasajeros", "fecha"]]
    print("Tabla reorganizada (serie, año, mes, pasajeros):")
    print(df[["serie", "anio", "mes", "pasajeros"]].head().to_string(index=False))
    return df


def split_train_test(df):
    section("PASO 3 - DIVISION ENTRENAMIENTO / PRUEBA")
    train = df[df["anio"] < ANIO_CORTE].copy()
    test = df[df["anio"] >= ANIO_CORTE].copy()
    print(f"Entrenamiento (año < {ANIO_CORTE}): {len(train)} filas "
          f"({train['fecha'].min():%Y-%m} a {train['fecha'].max():%Y-%m})")
    print(f"Prueba (año >= {ANIO_CORTE}): {len(test)} filas "
          f"({test['fecha'].min():%Y-%m} a {test['fecha'].max():%Y-%m})")
    return train, test


def design_matrix(frame, columns=None):
    dummies = pd.get_dummies(frame["mes"], prefix="mes", drop_first=True)
    matrix = pd.concat([frame[["serie"]].reset_index(drop=True),
                        dummies.reset_index(drop=True)], axis=1)
    if columns is not None:
        matrix = matrix.reindex(columns=columns, fill_value=0)
    return matrix


def metrics(y_true, y_pred):
    error = np.asarray(y_true) - np.asarray(y_pred)
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mape = float(np.mean(np.abs(error / np.asarray(y_true))) * 100)
    return mae, rmse, mape


def train_regression(train, test, future):
    section("PASO 4 - MODELO DE REGRESION (log-pasajeros ~ serie + mes)")
    X_train = design_matrix(train)
    y_train = np.log(train["pasajeros"].to_numpy())

    model = LinearRegression().fit(X_train, y_train)

    fitted = np.exp(model.predict(X_train))
    pred_test = np.exp(model.predict(design_matrix(test, X_train.columns)))
    pred_future = np.exp(model.predict(design_matrix(future, X_train.columns)))

    mae, rmse, mape = metrics(test["pasajeros"], pred_test)
    print(f"Coeficiente de la tendencia (serie): {model.coef_[0]:.5f} por mes en escala log")
    print(f"MAE en prueba : {mae:.2f} pasajeros")
    print(f"RMSE en prueba: {rmse:.2f} pasajeros")
    print(f"MAPE en prueba: {mape:.2f} %")
    return model, fitted, pred_test, pred_future, (mae, rmse, mape)


def train_holt_winters(train, test):
    section("PASO 5 - MODELO HOLT-WINTERS (ETS) Y PREVISION")
    serie_ts = pd.Series(
        train["pasajeros"].to_numpy(dtype=float),
        index=pd.date_range(start=train["fecha"].iloc[0], periods=len(train), freq="MS"),
    )
    model = ETSModel(serie_ts, error="add", trend="add",
                     seasonal="mul", seasonal_periods=SEASONAL_PERIODS)
    fit = model.fit(disp=False)

    steps = len(test) + HORIZONTE_FUTURO
    forecast = fit.get_prediction(start=len(train), end=len(train) + steps - 1)
    summary = forecast.summary_frame(alpha=0.05)

    fitted = fit.fittedvalues.to_numpy()
    mean = summary["mean"].to_numpy()
    low = summary["pi_lower"].to_numpy()
    high = summary["pi_upper"].to_numpy()

    mae, rmse, mape = metrics(test["pasajeros"], mean[:len(test)])
    print(f"Parametros suavizado -> nivel(alpha): {fit.params[0]:.3f}, "
          f"tendencia(beta): {fit.params[1]:.3f}, estacional(gamma): {fit.params[2]:.3f}")
    print(f"MAE en prueba : {mae:.2f} pasajeros")
    print(f"RMSE en prueba: {rmse:.2f} pasajeros")
    print(f"MAPE en prueba: {mape:.2f} %")
    return fit, fitted, mean, low, high, (mae, rmse, mape)


def future_frame(df):
    ultimo = df["serie"].max()
    inicio = df["fecha"].max() + pd.offsets.MonthBegin(1)
    fechas = pd.date_range(start=inicio, periods=HORIZONTE_FUTURO, freq="MS")
    return pd.DataFrame({
        "serie": np.arange(ultimo + 1, ultimo + 1 + HORIZONTE_FUTURO),
        "anio": fechas.year,
        "mes": fechas.month,
        "pasajeros": np.nan,
        "fecha": fechas,
    })


def build_export(df, train, test, future, reg_parts, hw_parts):
    section("PASO 6 - EXPORT DE RESULTADOS (CSV)")
    reg_fitted, reg_test, reg_future = reg_parts
    hw_fitted, hw_mean, hw_low, hw_high = hw_parts
    n_test = len(test)

    df = df.copy()
    df["conjunto"] = np.where(df["anio"] < ANIO_CORTE, "entrenamiento", "prueba")
    df["prediccion_hw"] = np.concatenate([hw_fitted, hw_mean[:n_test]])
    df["prediccion_reg"] = np.concatenate([reg_fitted, reg_test])
    df["pi_low"] = np.concatenate([np.full(len(train), np.nan), hw_low[:n_test]])
    df["pi_high"] = np.concatenate([np.full(len(train), np.nan), hw_high[:n_test]])

    fut = future.copy()
    fut["conjunto"] = "prevision"
    fut["prediccion_hw"] = hw_mean[n_test:]
    fut["prediccion_reg"] = reg_future
    fut["pi_low"] = hw_low[n_test:]
    fut["pi_high"] = hw_high[n_test:]

    columns = ["serie", "anio", "mes", "fecha", "pasajeros", "conjunto",
               "prediccion_hw", "pi_low", "pi_high", "prediccion_reg"]
    export = pd.concat([df[columns], fut[columns]], ignore_index=True)
    export = export.rename(columns={"anio": "año"})
    export["fecha"] = export["fecha"].dt.strftime("%Y-%m")
    for col in ["prediccion_hw", "pi_low", "pi_high", "prediccion_reg"]:
        export[col] = export[col].round(1)

    export.to_csv(EXPORT_PATH, index=False)
    print(f"CSV guardado en: {EXPORT_PATH.relative_to(BASE_DIR)} ({len(export)} filas)")
    print("\nUltimas filas (prevision futura):")
    print(export.tail(6).to_string(index=False))
    return export


def plot_series(df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df["fecha"], df["pasajeros"], color=COLOR_HW, linewidth=1.6)
    ax.set_title("Pasajeros por mes (serie completa)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Pasajeros")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "serie_completa.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Grafica guardada: {path.relative_to(BASE_DIR)}")


def plot_forecast(export):
    export = export.copy()
    export["fecha"] = pd.to_datetime(export["fecha"], format="%Y-%m")
    real = export[export["pasajeros"].notna()]
    corte = pd.Timestamp(f"{ANIO_CORTE}-01-01")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(real["fecha"], real["pasajeros"], color=COLOR_REAL,
            linewidth=1.8, label="Real")
    ax.plot(export["fecha"], export["prediccion_hw"], color=COLOR_HW,
            linewidth=1.6, label="Holt-Winters (previsión)")
    ax.plot(export["fecha"], export["prediccion_reg"], color=COLOR_REG,
            linewidth=1.2, linestyle="--", label="Regresión")
    banda = export[export["pi_low"].notna()]
    ax.fill_between(banda["fecha"], banda["pi_low"], banda["pi_high"],
                    color=COLOR_BANDA, alpha=0.35, label="Intervalo 95% (HW)")
    ax.axvline(corte, color=COLOR_GUIA, linestyle=":", linewidth=1.2)
    ax.text(corte, ax.get_ylim()[1] * 0.95, " prueba →", color=COLOR_GUIA, fontsize=9)
    ax.set_title("Previsión de pasajeros: real vs. modelos")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Pasajeros")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "prevision.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Grafica guardada: {path.relative_to(BASE_DIR)}")


def save_models(reg_model, hw_fit):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(reg_model, MODELS_DIR / "regresion_log.joblib")
    hw_fit.save(str(MODELS_DIR / "holt_winters_ets.joblib"))
    print(f"Modelos guardados en: {MODELS_DIR.relative_to(BASE_DIR)}")


def compare(reg_score, hw_score):
    section("PASO 7 - COMPARACION EN EL CONJUNTO DE PRUEBA (1960)")
    tabla = pd.DataFrame(
        [["Regresion (log)", *reg_score], ["Holt-Winters (ETS)", *hw_score]],
        columns=["modelo", "MAE", "RMSE", "MAPE_%"],
    ).round(2)
    print(tabla.to_string(index=False))


def main():
    df = load_dataset()
    df = build_columns(df)
    train, test = split_train_test(df)
    future = future_frame(df)

    reg_model, reg_fitted, reg_test, reg_future, reg_score = train_regression(train, test, future)
    hw_fit, hw_fitted, hw_mean, hw_low, hw_high, hw_score = train_holt_winters(train, test)

    export = build_export(
        df, train, test, future,
        (reg_fitted, reg_test, reg_future),
        (hw_fitted, hw_mean, hw_low, hw_high),
    )

    PLOTS_DIR.mkdir(exist_ok=True)
    plot_series(df)
    plot_forecast(export)
    save_models(reg_model, hw_fit)
    compare(reg_score, hw_score)


if __name__ == "__main__":
    main()
