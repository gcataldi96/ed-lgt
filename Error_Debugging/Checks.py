from simsio import logger


def pause(phrase, debug):
    if not isinstance(phrase, str):
        raise TypeError(f"phrase should be a STRING, not a {type(phrase)}")
    if not isinstance(debug, bool):
        raise TypeError(f"debug should be a BOOL, not a {type(debug)}")
    if debug == True:
        # IT PROVIDES A PAUSE (with a phrase) in a given point of the PYTHON CODE
        logger.info("----------------------------------------------------")
        # Press the <ENTER> key to continue
        programPause = input(phrase)
        logger.info("----------------------------------------------------")
        logger.info("")


def alert(phrase, debug):
    if not isinstance(phrase, str):
        raise TypeError(f"phrase should be a STRING, not a {type(phrase)}")
    if not isinstance(debug, bool):
        raise TypeError(f"debug should be a BOOL, not a {type(debug)}")
    if debug == True:
        # IT logger.infoS A PHRASE IN A GIVEN POINT OF A PYTHON CODE
        logger.info("")
        logger.info(phrase)
