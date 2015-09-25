library(dplyr)
library(ggplot2)
library(scales)
library(gridExtra)
library(corrplot) 
library(stargazer)
library(stringi)
library(caret)
library(caTools)

turnstile_data <- read.csv('turnstile_weather_v2.csv')
# turn date into POSIX to make easier to work with
turnstile_data$datetime <- as.POSIXct(turnstile_data$datetime)
# Add full name of the day of the week instead of day number
turnstile_data$day_week <- factor(format(turnstile_data$datetime, format = '%A'), 
                                  levels = c("Monday", "Tuesday", "Wednesday","Thursday",
                                             "Friday", "Saturday", "Sunday"))
# Also adding day number
turnstile_data$day_number <- as.numeric(turnstile_data$day_week)
# Adding formated hour with levels
turnstile_data$hour_format <- factor(format(turnstile_data$datetime, format = '%I:00 %p'),
                                     levels = c("04:00 AM", "08:00 AM","12:00 PM",
                                                "04:00 PM", "08:00 PM", "12:00 AM"))
# Adding number data for hour_format just created like day number
turnstile_data$hour_number <- as.numeric(turnstile_data$hour_format)
# Add strings to fog
turnstile_data$fog <- factor(turnstile_data$fog, levels=c(0,1), labels=c("No Fog", "Fog"))
# Add strings to rain
turnstile_data$rain <- factor(turnstile_data$rain, levels=c(0,1), labels=c("No Rain", "Rain"))
# Add strings to weekend / weekday
turnstile_data$weekday <- factor(turnstile_data$weekday, levels=c(0,1), labels=c("Weekend", "Weekday"))
# Combining date and day of week
turnstile_data$date_day_week <- factor(format(turnstile_data$datetime,format="%m-%d-%Y:%A"))
head(turnstile_data)


# TRAIN TEST SPLIT

#split = sample.split(turnstile_data$ENTRIESn_hourly, 0.7)
#train <- subset(turnstile_data, split == TRUE)
#test <- subset(turnstile_data, split == FALSE)
# Split the data into a 70 - 30 split so I can build with the train of 70% and test with the remaining 30% to evaluate my models
dim(train)
dim(test)

# separate only numeric numbers 
turn_numeric <- turnstile_data[,sapply(turnstile_data, is.numeric)]
# create correlation matrix
turnstile_data_corr_matrix <- cor(turn_numeric)
# plot 
corrplot(turnstile_data_corr_matrix, method = "ellipse", order="hclust", type="lower", 
         tl.cex=0.75, add = FALSE, tl.pos="lower") 
corrplot(turnstile_data_corr_matrix, method = "number", order="hclust", type="upper", 
         tl.cex=0.75, add = TRUE, tl.pos="upper")

# Using dplyr and the split apply combine functionality I can create day of week and entries for that day
Day_of_Week <- turnstile_data %>%
    group_by(day_week) %>%
    summarise(entries = sum(ENTRIESn_hourly))


# however the day of the week and counts are flawed since there is not an even amount of day in the month 
Day_of_Week_Filtered <- turnstile_data %>%
    group_by(date_day_week, day_week) %>%
    summarise(entries = sum(ENTRIESn_hourly)) %>%
    ungroup() %>%
    group_by(day_week) %>%
    mutate(total_days = n(),
           avg_entries_per_day = sum(entries) / mean(total_days)) %>%
    select(day_week, total_days, avg_entries_per_day) %>%
    distinct(day_week, total_days, avg_entries_per_day)

day_total <- ggplot(Day_of_Week, aes(x=day_week, y=entries, fill=day_week)) + 
    geom_bar(stat="identity") + 
    scale_fill_brewer(palette="Set2") +
    theme(axis.text.x = element_text(angle=45)) + 
    guides(fill=FALSE) + 
    scale_y_continuous(labels=comma) + 
    xlab("") + ylab("Total Entries") + ggtitle("Total Ridership for Each Day of the Week \n") + 
    geom_text(aes(x=day_week, y=500000, label= paste0("Total    \nDays: ", total_days)), size=3 , data=Day_of_Week_Filtered)

day_avg <- ggplot(Day_of_Week_Filtered, aes(x=day_week, y=avg_entries_per_day, fill=day_week)) + 
    geom_bar(stat="identity") + 
    scale_fill_brewer(palette="Set2") +
    theme(axis.text.x = element_text(angle=45)) + 
    guides(fill=FALSE) + 
    scale_y_continuous(labels=comma) + 
    xlab("") + ylab("Total Entries") + ggtitle("Average Ridership per Day \nfor Each Day of the Week") + 
    geom_text(aes(x=day_week, y=100000, label= paste0("Total    \nDays: ", total_days)), size=3 , data=Day_of_Week_Filtered)


grid.arrange(day_total, day_avg, ncol=2)

# Hours of the day
