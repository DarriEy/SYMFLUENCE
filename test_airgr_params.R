library(airGR)

# Create dummy data
DatesR <- seq(as.POSIXct("2015-01-01"), as.POSIXct("2015-12-31"), by = "day")
Precip <- rep(5, length(DatesR))
PotEvap <- rep(2, length(DatesR))
TempMean <- rep(10, length(DatesR))
ZInputs <- 100
HypsoData <- seq(0, 1000, length.out=101)

InputsModel <- CreateInputsModel(
    FUN_MOD = RunModel_CemaNeigeGR4J,
    DatesR = DatesR,
    Precip = Precip,
    PotEvap = PotEvap,
    TempMean = TempMean,
    HypsoData = HypsoData,
    ZInputs = ZInputs
)

RunOptions <- CreateRunOptions(
    FUN_MOD = RunModel_CemaNeigeGR4J,
    InputsModel = InputsModel,
    IndPeriod_WarmUp = 1:10,
    IndPeriod_Run = 11:365,
    IsHyst = TRUE
)

InputsCrit <- CreateInputsCrit(
    FUN_CRIT = ErrorCrit_KGE,
    InputsModel = InputsModel,
    RunOptions = RunOptions,
    Obs = rep(2, length(11:365)) # Dummy obs
)

CalibOptions <- CreateCalibOptions(
    FUN_MOD = RunModel_CemaNeigeGR4J,
    FUN_CALIB = Calibration_Michel,
    IsHyst = TRUE
)

cat("Number of parameters in SearchRanges:", nrow(CalibOptions$SearchRanges), "\n")
cat("SearchRanges:\n")
print(CalibOptions$SearchRanges)
