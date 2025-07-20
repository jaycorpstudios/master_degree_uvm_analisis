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

TukeyHSD(factorial_anova_model)

library(ggplot2)
library(dplyr)

summary_stats <- experiment_data %>%
  group_by(Treatment, Concentration) %>%
  summarise(
    mean_value = mean(Value),
    standard_error = sd(Value)/sqrt(n())
  )

ggplot(summary_stats, aes(x = Treatment, y = mean_value, fill = Concentration)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_value - standard_error, ymax = mean_value + standard_error), 
                width = 0.2, position = position_dodge(0.9)) +
  labs(title = "Mean Values by Treatment and Concentration",
       y = "Average Value",
       x = "Treatment") +
  theme_minimal()

