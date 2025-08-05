rm(list = ls())
gcol <- gc()

suppressPackageStartupMessages({
  library(dplyr)
  library(magrittr)
  library(jsonlite)
  library(purrr)
  library(progressr)
  library(data.table)
  library(glue)
  library(optparse)
  library(furrr)
  library(cli)
})

option_list <- list(
  make_option(c("-s", "--start_year"),
              action = "store",
              default = wehoop:::most_recent_wnba_season(),
              type = "integer",
              help = "Start year of the seasons to process"),
  make_option(c("-e", "--end_year"),
              action = "store",
              default = wehoop:::most_recent_wnba_season(),
              type = "integer",
              help = "End year of the seasons to process")
)

opt <- parse_args(OptionParser(option_list = option_list))
options(stringsAsFactors = FALSE)
options(scipen = 999)
years_vec <- opt$s:opt$e


wnba_player_box_games <- function(y) {
  espn_df <- data.frame()
  sched <- readRDS(paste0("wnba/schedules/rds/wnba_schedule_", y, ".rds"))

  season_player_box_list <- sched %>%
    filter(.data$game_json == TRUE) %>%
    pull("game_id")

  if (length(season_player_box_list) > 0) {
    cli::cli_progress_step(msg = "Compiling {y} ESPN WNBA Player Boxscores ({length(season_player_box_list)} games)",
                           msg_done = "Compiled {y} ESPN WNBA Player Boxscores!")

    future::plan("multisession")
    espn_df <- furrr::future_map_dfr(season_player_box_list, function(x) {
      tryCatch(
        expr = {
          resp <- glue::glue("https://raw.githubusercontent.com/sportsdataverse/wehoop-wnba-raw/main/wnba/json/final/{x}.json")
          player_box_score <- wehoop:::helper_espn_wnba_player_box(resp)
          return(player_box_score)
        },
        error = function(e) {
          message(glue::glue("{Sys.time()}: Player box score data for {x} issue!"))
          NULL
        }
      )
    }, .options = furrr::furrr_options(seed = TRUE))
  }

  if (nrow(espn_df) > 1) {
    espn_df <- espn_df %>%
      arrange(desc(.data$game_date)) %>%
      wehoop:::make_wehoop_data("ESPN WNBA Player Boxscores from wehoop data repository", Sys.time())

    ifelse(!dir.exists(file.path("wnba/player_box")), dir.create(file.path("wnba/player_box")), FALSE)

    data.table::fwrite(espn_df, file = paste0("wnba/player_box/player_box_", y, ".csv"))

    cli::cli_progress_step(msg = "Saved PlayerBox CSV for {y}",
                           msg_done = "PlayerBox CSV {y} saved successfully!")
  } else {
    cli::cli_alert_info("No PlayerBox data to compile for {y}")
  }

  rm(espn_df)
  gc()
  return(NULL)
}



purrr::map(years_vec, function(y) {
  wnba_player_box_games(y)
  return(NULL)
})

rm(years_vec)
gcol <- gc()
