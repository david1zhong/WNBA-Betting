rm(list = ls())
gc()

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
  library(arrow)
  library(lubridate)
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
  sched <- readRDS(glue("wnba/schedules/rds/wnba_schedule_{y}.rds"))

  season_player_box_list <- sched %>%
    filter(game_json == TRUE) %>%
    pull("game_id")

  if (length(season_player_box_list) > 0) {
    cli::cli_progress_step(
      msg = "Compiling {y} ESPN WNBA Player Boxscores ({length(season_player_box_list)} games)",
      msg_done = "Compiled {y} ESPN WNBA Player Boxscores!"
    )

    future::plan("multisession")
    espn_df <- furrr::future_map_dfr(season_player_box_list, function(x) {
      tryCatch({
        resp <- glue("https://raw.githubusercontent.com/sportsdataverse/wehoop-wnba-raw/main/wnba/json/final/{x}.json")
        wehoop:::helper_espn_wnba_player_box(resp)
      }, error = function(e) {
        message(glue("{Sys.time()}: Player box score data for {x} issue!"))
        NULL
      })
    }, .options = furrr::furrr_options(seed = TRUE))
  }

  if (nrow(espn_df) > 1) {
    espn_df <- espn_df %>%
      arrange(desc(game_date)) %>%
      wehoop:::make_wehoop_data("ESPN WNBA Player Boxscores from wehoop data repository", Sys.time())

    dir.create("wnba/player_box/csv", recursive = TRUE, showWarnings = FALSE)
    data.table::fwrite(espn_df, glue("wnba/player_box/csv/player_box_{y}.csv"))

    cli::cli_progress_step(
      msg = "Saved PlayerBox CSV for {y}",
      msg_done = "PlayerBox CSV {y} saved successfully!"
    )
  } else {
    cli::cli_alert_info("No PlayerBox data to compile for {y}")
  }

  final_sched <- sched %>%
    distinct() %>%
    arrange(desc(date)) %>%
    wehoop:::make_wehoop_data("ESPN WNBA Schedule from wehoop data repository", Sys.time())

  dir.create("wnba/schedules/rds", recursive = TRUE, showWarnings = FALSE)
  saveRDS(final_sched, glue("wnba/schedules/rds/wnba_schedule_{y}.rds"))

  rm(espn_df, sched, final_sched)
  gc()
  return(NULL)
}

purrr::map(years_vec, wnba_player_box_games)

sched_list <- list.files(path = "wnba/schedules/rds/", pattern = "wnba_schedule_.*\\.rds$")
sched_g <- purrr::map_dfr(sched_list, function(x) {
  readRDS(file.path("wnba/schedules/rds", x)) %>%
    mutate(across(any_of(c(
      "id", "game_id", "type_id", "status_type_id", "home_id",
      "home_venue_id", "home_conference_id", "home_score",
      "away_id", "away_venue_id", "away_conference_id",
      "away_score", "season", "season_type", "groups_id",
      "tournament_id", "venue_id"
    )), as.integer)) %>%
    mutate(
      status_display_clock = as.character(status_display_clock),
      game_date_time = ymd_hm(substr(date, 1, nchar(date) - 1)) %>%
        with_tz(tzone = "America/New_York"),
      game_date = as.Date(substr(game_date_time, 1, 10))
    )
})

final_sched <- sched_g %>%
  wehoop:::make_wehoop_data("ESPN WNBA Master Schedule from wehoop data repository", Sys.time()) %>%
  arrange(desc(date))

saveRDS(final_sched, "wnba/schedules/rds/wnba_schedule_master.rds")

rm(years_vec, sched_g, final_sched, sched_list)
gc()
