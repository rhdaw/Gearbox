from flowsom_objects import PipelineConfig, FlowSOMPipeline
import numpy as np


def main():
    config = PipelineConfig(
        unitogated_csv_dir="/Users/user/Documents/Gearbox/3_unito/UNITO_csv_conversion_predict",  # Where the post UNITO csv files are located
        csv_dir_metadir="/Users/user/Documents/Gearbox/3_unito/UNITO_csv_conversion/metadata",  # The same directory as csv_conversion_dir_metadir in unito's main.py()
        filtered_fcs_path="/Users/user/Documents/Gearbox/4_flowsom/flowsom_full/filtered_fcs_files",  # Where you want the filtered fcs files to go
        filter_out=["Neutrophils_pred"],
        marker_list=[
            "CD45",
            "CD33",
            "CD4",
            "CD16",
            "CD14",
            "CX3CR1",
            "CD27",
            "CCR6",
            "CD62L",
            "CCR2",
            "CD25",
            "CD8",
            "CD32",
            "CD86",
            "CD64",
            "TCRGD",
            "CD15",
            "CD28",
            "CD36",
            "CCR5",
            "CD45RA",
            "CD163",
            "FCE1RA",
            "CD56",
            "CD123",
            "CD19",
            "CCR7",
            "CD3",
            "HLADR",
        ],  # List of the markers in your csv files that you want to use for FlowSOM clustering
        cluster_num=30,  # Number of clusters created
        seed=42,
        custom_threshold_dict={
            "CD45": 2.587410397350002,
            "CD33": 3.2036517649841896,
            "CD4": 2.7318646200929146,
            "CD16": 3.849351394057948,
            "CX3CR1": 3.848135779655888,
            "CD14": 3.6502597439184514,
            "CD27": 4.624077500386524,
            "CCR6": 4.77491585585376,
            "CD62L": 4.888211298737665,
            "CCR2": 2.9391349417271813,
            "CD25": 3.6373366582162907,
            "CD8": 2.846675332383487,
            "CD32": 3.3563190794197495,
            "CD86": 4.8710552825944715,
            "CD64": 4.931486314714817,
            "TCRgd": 4.252128898258721,
            "CD15": 3.1599065425561443,
            "CD28": 4.440044040676711,
            "CD36": 2.5932030729567117,
            "CCR5": 5.181820692418106,
            "CD45RA": 4.37399967013087,
            "CD163": 4.239593442344301,
            "FCE1RA": 4.975021958029761,
            "CD56": 4.936193367952034,
            "CD123": 5.15529079854538,
            "CD19": 4.4965485529182265,
            "CCR7": 5.14647175470327,
            "CD3": 3.8657924282437053,
            "HLADR": 3.7367705531196025,
            "pSTAT5": 4.060106458520733,
            "pCREB": 3.474551957035929,
            "pSTAT3Y705": 4.327328078922303,
            "pP38": 4.877525454098063,
            "pNFKB": 4.925729673282306,
            "pSTAT1": 5.163742223709236,
            "pSMAD23": 5.240082461929855,
            "pSTAT3_S727": 3.925978828866184,
            "pAKT": 5.4537709062305195,
        },
    )

    # Run pipeline
    pipeline = FlowSOMPipeline(config)
    # fsom = pipeline.run(som_xdim=10, som_ydim=10)

    # pkl_path = (
    #     "/Users/user/Documents/Gearbox/4_flowsom/flowsom_full/flowsom_full_object.pkl"
    # )
    pkl_path_2 = "/Users/user/Documents/Gearbox/4_flowsom/flowsom_full/flowsom_full_object_umap.pkl"
    # pipeline.save_flowsom_pkl(fsom, pkl_path)

    fsom = pipeline.load_flowsom_pkl(pkl_path_2)
    # p = pipeline.plot_flowsom(
    #     fsom, "/Users/user/Documents/Gearbox/4_flowsom/flowsom_full/"
    # )

    # umap_plot = pipeline.plot_umap(
    #     fsom,
    #     "/Users/user/Documents/Gearbox/4_flowsom/flowsom_full/",
    #     markers=[
    #         "CD45",
    #         "CD33",
    #         "CD4",
    #         "CD16",
    #         "CD14",
    #         "CX3CR1",
    #         "CD27",
    #         "CCR6",
    #         "CD62L",
    #         "CCR2",
    #         "CD25",
    #         "CD8",
    #         "CD32",
    #         "CD86",
    #         "CD64",
    #         "TCRGD",
    #         "CD15",
    #         "CD28",
    #         "CD36",
    #         "CCR5",
    #         "CD45RA",
    #         "CD163",
    #         "FCE1RA",
    #         "CD56",
    #         "CD123",
    #         "CD19",
    #         "CCR7",
    #         "CD3",
    #         "HLADR",
    #         "pCREB",
    #         "pSTAT3_Y705",
    #         "pP38",
    #         "pNFKB",
    #         "pSTAT1",
    #         "pSMAD2-3",
    #         "pSTAT3_S727",
    #         "pAKT",
    #     ],
    #     pkl_post_umap_path=pkl_path_2,
    #     plot_subsample_n=15_000_000,
    # )

    # pipeline.save_readouts(
    #     fsom,
    #     "/Users/user/Documents/Gearbox/4_flowsom/flowsom_full/add/",
    #     "metaclusters",
    #     threshold_method="custom",
    #     threshold_report=True,
    # )

    pipeline.plot_umap_save_readouts_by_fcsfile(
        fsom,
        "/Users/user/Documents/Gearbox/4_flowsom/flowsom_full/",
        markers=np.array(config.marker_list),  # must be np.array for this method
        threshold_method="custom",
    )


if __name__ == "__main__":
    main()
