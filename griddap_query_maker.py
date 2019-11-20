from urllib import error, request
import sys
import csv

""" This script reads a longitude and latitude range from a file called params.csv
 (which must be in the same folder as the script), and builds a url to query the 
 jplMURSST41 dataset at https://coastwatch.pfeg.noaa.gov/erddap/griddap/ and the
 data is downloaded to a csv file.
 Each line of params.csv creates a separate request and will results in a separate
 csv file.
 By default the script gets the data for the entire time period data is available 
 for"""

def make_request(params_dict):

    # parameters that have been left blank in params.csv need to be filled in
    # with defaults

    # This is a required field - script exits if blank
    if params_dict["lat_start"] == '':
        print("\nPlease specify latstart - exiting")
        exit(0)

    if params_dict["lat_stride"] == '':
        params_dict["lat_stride"] = 1

    # This is a required field - script exits if blank
    if params_dict["lat_stop"] == '':
        print("\nPlease specify latstop - exiting")
        exit(0)

    # This is a required field - script exits if blank
    if params_dict["long_start"] == '':
        print("\nPlease specify longstart - exiting")
        exit(0)

    if params_dict["long_stride"] == '':
        params_dict["long_stride"] = 1

    # This is a required field - script exits if blank
    if params_dict["long_stop"] == '':
        print("\nPlease specify longstop - exiting")
        exit(0)

    if params_dict["datasetID"] == '':
        params_dict["datasetID"] = "jplMURSST41"

    if params_dict["fileType"] == '':
        params_dict["fileType"] = ".csv"

    if params_dict["dataVariableName"] == '':
        params_dict["dataVariableName"] = "analysed_sst"

    # This is a required field - script exits if blank
    if params_dict["startDateTime"] == '':
        print("\nPlease specify date_start - exiting")
        exit(0)

    # if no time stride has been specified, default to 1 which gets every data
    # point
    if params_dict["strideDateTime"] == '':
        params_dict["strideDateTime"] = 1

    # if stopDateTime has been left blank, that is fine, we just get all the
    # data from start date only

    # if outputFileName has been left blank, request a filename from the user
    if params_dict["output_filename"] == '':
        outputFileName = input("\nPlease specify a filename for this data download: ")
    else:
        outputFileName = params_dict["output_filename"]
    if not outputFileName.endswith(".csv"):
        outputFileName = outputFileName + ".csv"



    query_url = construct_url(params_dict)
    print("\nRequesting data from: ")
    print(query_url)
    print("\nData will be saved to " + outputFileName)

    try:
        request.urlretrieve(query_url, outputFileName)
    except error.HTTPError:
        print("\n **Problem with params.csv. Please check the parameters you've entered.**")
        exit(0)




def construct_url(params):

    url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/" \
        + params["datasetID"] \
        + params["fileType"] \
        + "?" \
        + params["dataVariableName"] \
        + "["

    if params["startDateTime"]:
        url = url + "(" + params["startDateTime"] + ")"
    if params["stopDateTime"]:
        url = url + ":" + str(params["strideDateTime"]) \
            + ":(" + str(params["stopDateTime"]) + ")"

    url = url + "][(" \
        + str(params["lat_start"]) + "):" \
        + str(params["lat_stride"]) + ":(" \
        + str(params["lat_stop"]) + ")][(" \
        + str(params["long_start"]) + "):" \
        + str(params["long_stride"]) + ":(" \
        + str(params["long_stop"]) + ")]"

    return url



def main(param_file):

    # open the parameter csv file into a csvreader object
    with open(param_file, 'r') as f:
        reader = csv.DictReader(f, fieldnames=["lat_start",
                                               "lat_stride",
                                               "lat_stop",
                                               "long_start",
                                               "long_stride",
                                               "long_stop",
                                               "datasetID",
                                               "fileType",
                                               "dataVariableName",
                                               "startDateTime",
                                               "strideDateTime",
                                               "stopDateTime",
                                               "output_filename"])
        # TODO: have it check that the first row really is a header before just skipping it automatically
        # skip the header
        next(reader)

        # for each line in the file, make a request
        try:
            for line in reader:
                make_request(line)
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (param_file, reader.line_num, e))




if __name__ == '__main__':

    try:
        param_file = sys.argv[1]
    except IndexError:
        param_file = "params.csv"

    main(param_file)
