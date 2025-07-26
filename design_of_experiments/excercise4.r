
solvent <- factor(rep(1:5, times = 4))
press <- factor(rep(c("A", "B", "C", "D"), each = 5))
yield <- c(
  87, 86, 88, 83,   # Solvent 1
  85, 87, 95, 85,   # Solvent 2
  90, 92, 95, 90,   # Solvent 3
  88, 97, 98, 88,   # Solvent 4
  99, 96, 91, 90    # Solvent 5
)

data <- data.frame(Solvent = solvent, Press = press, Yield = yield)


# Visualize how the response variable varies across treatments and concentrations
library(ggplot2)

# Opción 1: Gráfico de puntos con líneas de interacción
ggplot(data, aes(x = Solvent, y = Yield, color = Press, group = Press)) +
  geom_point(size = 3) +
  geom_line() +
  labs(title = "Oil yield by Solvent and Press",
       x = "Solvent",
       y = "Yield (ml)") +
  theme_minimal()

# Opción 2: Gráfico de barras agrupadas
ggplot(data, aes(x = Solvent, y = Yield, fill = Press)) +
  geom_col(position = "dodge", alpha = 0.8) +
  labs(title = "Oil yield by Solvent and Press",
       x = "Solvent",
       y = "Yield (ml)") +
  theme_minimal()

# Opción 3: Facet wrap para separar por Press
ggplot(data, aes(x = Solvent, y = Yield, fill = Press)) +
  geom_col() +
  facet_wrap(~Press, ncol = 2) +
  labs(title = "Oil yield by Solvent and Press",
       x = "Solvent",
       y = "Yield (ml)") +
  theme_minimal() +
  theme(legend.position = "none")

anova_model <- aov(Yield ~ Solvent + Press, data = data)
summary(anova_model)

# Post-hoc analysis using Tukey's HSD test as seen in previous class
tukey_result <- TukeyHSD(anova_model, "Solvent")
print(tukey_result)

# Residuals normality test
shapiro.test(resid(anova_model))
