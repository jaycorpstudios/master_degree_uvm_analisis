treatment <- factor(rep(1:3, each = 6))
concentration <- factor(rep(rep(1:2, each = 3), times = 3))
response_values <- c(
  9.8, 10.1, 9.8, 11.3, 10.7, 10.7,  # Treatment 1
  9.2, 8.6, 9.2, 10.3, 10.7, 10.2,   # Treatment 2
  8.4, 7.9, 8.0, 9.8, 10.1, 10.1     # Treatment 3
)

experiment_data <- data.frame(Treatment = treatment, Concentration = concentration, Value = response_values)


factorial_anova_model <- aov(Value ~ Treatment * Concentration, data = experiment_data)
summary(factorial_anova_model)

shapiro.test(resid(factorial_anova_model))

library(ggplot2)

ggplot(experiment_data, aes(x = Treatment, y = response_values, fill = Concentration)) +
  geom_boxplot() +
  labs(title = "Boxplot of Response Values by Treatment and Concentration",
       y = "Sample Value",
       x = "Treatment") +
  theme_minimal()

