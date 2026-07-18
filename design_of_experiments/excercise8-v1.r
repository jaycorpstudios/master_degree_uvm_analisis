# 1. Load experimental data
cat("\n========== LOADING EXPERIMENTAL DATA ==========\n")
data <- data.frame(
  A = c(
    -1,  1, -1,  1, -1,  1, -1,  1,
    -1,  1, -1,  1, -1,  1, -1,  1,
    -1,  1, -1,  1, -1,  1, -1,  1,
    -1,  1, -1,  1, -1,  1, -1,  1,
     0,  0,  0,  0
  ),
  B = c(
    -1, -1,  1,  1, -1, -1,  1,  1,
    -1, -1,  1,  1, -1, -1,  1,  1,
    -1, -1,  1,  1, -1, -1,  1,  1,
    -1, -1,  1,  1, -1, -1,  1,  1,
     0,  0,  0,  0
  ),
  C = c(
    -1, -1, -1, -1,  1,  1,  1,  1,
    -1, -1, -1, -1,  1,  1,  1,  1,
    -1, -1, -1, -1,  1,  1,  1,  1,
    -1, -1, -1, -1,  1,  1,  1,  1,
     0,  0,  0,  0
  ),
  D = c(
    -1, -1, -1, -1, -1, -1, -1, -1,
     1,  1,  1,  1,  1,  1,  1,  1,
    -1, -1, -1, -1, -1, -1, -1, -1,
     1,  1,  1,  1,  1,  1,  1,  1,
     0,  0,  0,  0
  ),
  E = c(
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
     1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,
     0,  0,  0,  0
  ),
  Y = c(
    1123, 1786, 1786, 1359,  982, 1458, 1451, 2180,
    3348, 3055, 2509, 2917, 4328, 3969, 2932, 4167,
    4093, 4517, 4755, 4316, 7066, 5871, 5477, 5774,
    4190, 4413, 4264, 4100, 6935, 6467, 5306, 5960,
    5134, 5157, 4653, 4834
  )
)
cat("Data loaded successfully. Total observations:", nrow(data), "\n")

# 2. Fit first-order model
cat("\n========== FITTING FIRST-ORDER MODEL ==========\n")
first_order_model <- lm(Y ~ A + B + C + D + E, data = data)
summary(first_order_model)


# =======================================================

# 3bis. Prueba formal de falta de ajuste usando puntos centrales
cat("\n========== FORMAL LACK OF FIT TEST ==========\n")

# Creamos un factor que identifica si el punto es central o factorial
data$tipo <- ifelse(rowSums(abs(data[, c("A","B","C","D","E")])) == 0, "central", "factorial")

# Ajustamos el modelo incluyendo el tipo para estimar el error puro (réplicas)
model_lackfit <- lm(Y ~ A + B + C + D + E, data = data)
anova_lackfit <- anova(model_lackfit)

# Ahora calculamos la suma de cuadrados de falta de ajuste y error puro

# Suma de cuadrados del error total (residual del modelo)
SSE <- sum(resid(model_lackfit)^2)

# Suma de cuadrados del error puro (variación dentro de los puntos centrales)
pure_error <- var(data$Y[data$tipo == "central"]) * (length(data$Y[data$tipo == "central"]) - 1)

# Grados de libertad del error puro
df_pure_error <- length(data$Y[data$tipo == "central"]) - 1

# Suma de cuadrados de falta de ajuste
SSLOF <- SSE - pure_error

# Grados de libertad de falta de ajuste
df_LOF <- model_lackfit$df.residual - df_pure_error

# Estadístico F para falta de ajuste
F_LOF <- (SSLOF / df_LOF) / (pure_error / df_pure_error)

# Valor p
p_LOF <- pf(F_LOF, df_LOF, df_pure_error, lower.tail = FALSE)

cat("Suma de cuadrados falta de ajuste:", SSLOF, "\n")
cat("Grados de libertad falta de ajuste:", df_LOF, "\n")
cat("Suma de cuadrados error puro:", pure_error, "\n")
cat("Grados de libertad error puro:", df_pure_error, "\n")
cat("Estadístico F falta de ajuste:", F_LOF, "\n")
cat("Valor p falta de ajuste:", p_LOF, "\n")

if (p_LOF < 0.05) {
  cat("=> Existe evidencia significativa de falta de ajuste. Modelo de primer orden NO es adecuado.\n")
} else {
  cat("=> No existe evidencia significativa de falta de ajuste. Modelo de primer orden adecuado.\n")
}


# =======================================================


# 3. Lack of fit test
cat("\n========== LACK OF FIT TEST ==========\n")
library(car)
cat("Running ANOVA to check global significance...\n")
Anova(first_order_model, type="II")


# 4. Calculate steepest ascent direction
cat("\n========== STEEPEST ASCENT DIRECTION ==========\n")
coefficients <- coef(first_order_model)[-1] # exclude intercept
base_step <- 1 / max(abs(coefficients))
direction <- coefficients * base_step
cat("Model coefficients (excluding intercept):\n")
print(coefficients)
cat("\nSteepest ascent direction:\n")
print(direction)


# 5. Generate steepest ascent points
cat("\n========== GENERATING STEEPEST ASCENT POINTS ==========\n")
steepest_points <- data.frame(
  A = 0 + direction["A"] * 1:4,
  B = 0 + direction["B"] * 1:4,
  C = 0 + direction["C"] * 1:4,
  D = 0 + direction["D"] * 1:4,
  E = 0 + direction["E"] * 1:4
)
cat("Steepest ascent points:\n")
print(steepest_points)


# 6. Plot steepest ascent path
cat("\n========== PLOTTING STEEPEST ASCENT PATH ==========\n")
library(ggplot2)
plot <- ggplot(steepest_points, aes(x = A, y = B)) +
  geom_point(size=3, color="blue") +
  geom_path() +
  theme_minimal() +
  labs(title="Steepest Ascent Direction (A vs B Projection)")
print(plot)
cat("Plot generated successfully.\n")