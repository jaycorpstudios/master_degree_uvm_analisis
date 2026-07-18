

data <- data.frame (
  Antioxidant = rep(c("Control", "A", "B", "C", "D"), each =6),
  Time = rep(rep(c(4, 8, 12), each =2), 5),
  UFC = c (
    3.84, 3.72, 27.63, 27.58, 39.95, 39.00, #Control
    4.00, 3.91, 22.00, 21.83, 46.20, 45.60, #A
    3.61, 3.61, 21.94, 21.85, 46.58, 42.98, #B
    3.57, 3.50, 20.50, 20.32, 45.14, 44.89, #C
    3.64, 3.61, 20.30, 20.19, 44.36, 44.02  #D
  )
)


library(ggplot2)
# Graph to visualize the data
ggplot(data, aes(x = Time, y = UFC, color = Antioxidant, group = Antioxidant)) +
  stat_summary(fun = mean, geom = "line", size = 1.2) +
  stat_summary(fun = mean, geom = "point", size = 3) +
  theme_minimal() +
  labs(title = "ANTIOXIDANT",
       x = "Time (Hours)",
       y = "Peroxide Units (UFC)")

library(multcomp)

data$Antioxidant <- factor(data$Antioxidant)
data$Antioxidant <- relevel(data$Antioxidant, ref = "Control")
data$Time <- as.factor(data$Time)

# ANOVA calculation
model <- aov(UFC ~ Antioxidant + Time, data = data)
summary(model)

# Tukey test
TukeyHSD(model, "Antioxidant")

# Dunnett comparison
dunnett_comparison <- glht(model, linfct = mcp(Antioxidant = "Dunnett"))
summary(dunnett_comparison)

