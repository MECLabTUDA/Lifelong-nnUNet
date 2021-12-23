#########################################################################################################
#-----------This class represents the Parameter Searcj of networks using the extended nnUNet------------#
#-----------                            trainer version.                                    ------------#
#########################################################################################################

from nnunet_ext.paths import param_search_output_dir
from nnunet_ext.parameter_search.param_searcher import ParamSearcher

def run_param_search():
    # -- First of all check that evaluation_output_dir is set otherwise we do not perform an evaluation -- #
    assert param_search_output_dir is not None, "Before running any parameter search, please specify the Parameter Search folder (PARAM_SEARCH_FOLDER) as described in the paths.md."

    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add standard arguments -- #
    
    # ----------------------------------------------
    # Perform parameter search based on users input
    # ----------------------------------------------
    searcher = ParamSearcher()
    searcher.do_search()


if __name__ == "__main__":
    run_param_search()