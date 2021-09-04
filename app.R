


library(fpp2)
library(shiny)

country_France <- read.csv('GlobalLandTemperaturesFranceAvg.csv')
country_Canada <- read.csv('GlobalLandTemperaturesCanadaAvg.csv')
country_Kazakstan <- read.csv('GlobalLandTemperaturesKazakAvg.csv')
country_NewZealand <- read.csv('GlobalLandTemperaturesNewZealAvg.csv')
country_Peru <- read.csv('GlobalLandTemperaturesPeruAvg.csv')
country_SouthAfrica <- read.csv('GlobalLandTemperaturesSouthAfAvg.csv')
country_SouthKorea <- read.csv('GlobalLandTemperaturesSouthKorAvg.csv')

country_France_TS <- ts(country_France, start = 1980, end = 2014, frequency = 12)
country_France_TS <- country_France_TS[,2]

country_Canada_TS <- ts(country_Canada, start = 1980, end = 2014, frequency = 12)
country_Canada_TS <- country_Canada_TS[,2]

country_Kazakstan_TS <- ts(country_Kazakstan, start = 1980, end = 2014, frequency = 12)
country_Kazakstan_TS <- country_Kazakstan_TS[,2]

country_Peru_TS <- ts(country_Peru, start = 1980
                      , end = 2014, frequency = 12)
country_Peru_TS <- country_Peru_TS[,2]

country_SouthAfrica_TS <- ts(country_SouthAfrica, start = 1980, end = 2014, frequency = 12)
country_SouthAfrica_TS <- country_SouthAfrica_TS[,2]

country_SouthKorea_TS <- ts(country_SouthKorea, start = 1980, end = 2014, frequency = 12)
country_SouthKorea_TS <- country_SouthKorea_TS[,2]


hwSmoothFrance_TS <- hw(country_France_TS, seasonal = "additive")
francefitDamped <- ets(country_France_TS, model = "ZZZ", damped = TRUE, additive.only = TRUE)
francefitNotDamped <- ets(country_France_TS, model = "ZZZ", damped = FALSE, additive.only = TRUE)

hwSmoothCanada_TS <- hw(country_Canada_TS, seasonal = "additive")
canadafitDamped <- ets(country_Canada_TS, model = "ZZZ", damped = TRUE, additive.only = TRUE)
canadafitNotDamped <- ets(country_Canada_TS, model = "ZZZ", damped = FALSE, additive.only = TRUE)

hwSmoothKazakstan_TS <- hw(country_Kazakstan_TS, seasonal = "additive")
kazakfitDamped <- ets(country_Kazakstan_TS, model = "ZZZ", damped = TRUE, additive.only = TRUE)
kazakfitNotDamped <- ets(country_Kazakstan_TS, model = "ZZZ", damped = FALSE, additive.only = TRUE)

hwSmoothPeru_TS <- hw(country_Peru_TS, seasonal = "additive")
perufitDamped <- ets(country_Peru_TS, model = "ZZZ", damped = TRUE, additive.only = TRUE)
perufitNotDamped <- ets(country_Peru_TS, model = "ZZZ", damped = FALSE, additive.only = TRUE)

hwSmoothSouthAfrica_TS <- hw(country_SouthAfrica_TS, seasonal = "additive")
southAfricafitDamped <- ets(country_SouthAfrica_TS, model = "ZZZ", 
                            damped = TRUE, additive.only = TRUE)
southAfricafitNotDamped <- ets(country_SouthAfrica_TS, model = "ZZZ", 
                               damped = FALSE, additive.only = TRUE)

hwSmoothSouthKorea_TS <- hw(country_SouthKorea_TS, seasonal = "additive")
southKoreafitDamped <- ets(country_SouthKorea_TS, model = "ZZZ", 
                           damped = TRUE, additive.only = TRUE)
southKoreafitNotDamped <- ets(country_SouthKorea_TS, model = "ZZZ", 
                              damped = FALSE, additive.only = TRUE)


## Only run this example in interactive R sessions
if (interactive()) {
    # pass a callback function to DataTables using I()
    shinyApp(
        ui = fluidPage(
            fluidRow(
                column(12,
                       dataTableOutput('table')
                )
            )
        ),
        server = function(input, output) {
            output$table <- renderDataTable(country_France_TS,
                                options = list(
                                pageLength = 5,
                                initComplete = I("function(settings, json) {alert('Done.');}")
                                            )
            )
        }
    )
}
