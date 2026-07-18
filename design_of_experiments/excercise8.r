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

## Iteration as first-order model was not adequate

cat("\n========== FITTING SECOND-ORDER MODEL ==========\n")

second_order_model <- lm(Y ~ (A + B + C + D + E)^2 + I(A^2) + I(B^2) + I(C^2) + I(D^2) + I(E^2), data = data)
summary(second_order_model)

cat("\n========== ANOVA SECOND-ORDER MODEL ==========\n")
anova(second_order_model)

cat("\n========== LACK OF FIT TEST SECOND-ORDER MODEL ==========\n")
# Para falta de ajuste en modelos más complejos, se debe tener cuidado:
# Aquí simplificamos con un test basado en residuos y réplica central
# O podemos comparar residuos con variabilidad en centro
library(car)
Anova(second_order_model, type="II")


cat("\n========== GRADIENT OF SECOND-ORDER MODEL ==========\n")

coefs <- coef(second_order_model)
linear_coefs <- coefs[c("A", "B", "C", "D", "E")]

base_step <- 1 / max(abs(linear_coefs))
direction <- linear_coefs * base_step

cat("Linear coefficients:\n")
print(linear_coefs)

cat("\nNormalized direction for steepest ascent:\n")
print(direction)

cat("\n========== GENERATING POINTS IN DIRECTION ==========\n")

steps <- 1:4
ascent_points <- data.frame(
  A = direction["A"] * steps,
  B = direction["B"] * steps,
  C = direction["C"] * steps,
  D = direction["D"] * steps,
  E = direction["E"] * steps
)

print(ascent_points)


# Finallly lets visualize the results

library(ggplot2)
library(reshape2)

ascent_points_plot <- rbind(
  data.frame(A = 0, B = 0, C = 0, D = 0, E = 0),
  ascent_points
)

ascent_points_plot$Step <- 0:(nrow(ascent_points_plot)-1)


df_long <- melt(ascent_points_plot, id.vars = "Step", variable.name = "Factor", value.name = "Value")


ggplot(df_long, aes(x = Step, y = Value, color = Factor)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(
    title = "Steepest Ascent Direction (Coded Factors)",
    x = "Step in Optimal Direction",
    y = "Coded Level"
  )


