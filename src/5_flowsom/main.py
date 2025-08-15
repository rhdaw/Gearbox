from flowsom_objects import PipelineConfig, FlowSOMPipeline
import numpy as np

def main():

    config = PipelineConfig(
    unitogated_csv_dir = '/Users/user/Documents/UNITO_csv_conversion/train/', # Where the post UNITO csv files are located
    csv_dir_metadir = '/Users/user/Documents/UNITO_csv_conversion/metadata/', # The same directory as csv_conversion_dir_metadir in unito's main.py()
    filtered_fcs_path = '/Users/user/Documents/UNITO_csv_conversion/flowsomtest/', # Where you want the filtered fcs files to go
    filter_out = ['Neutrophils'],
    marker_list = [
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
        "HLADR"
    ], # List of the markers in your csv files that you want to use for FlowSOM clustering
    cluster_num = 25, # Number of clusters created
    seed = 42
    )

    # Run pipeline
    pipeline = FlowSOMPipeline(config)
    fsom = pipeline.run(som_xdim=10, som_ydim=10)

    # p = pipeline.plot_flowsom(fsom,
    #                         '/Users/user/Documents/UNITO_csv_conversion/flowsomtest/'
    #                         )

    # umap_plot = pipeline.plot_umap(fsom,
    #                     '/Users/user/Documents/UNITO_csv_conversion/flowsomtest/',
    #                     subsample=True,
    #                     n_sample = 100_000
    #                     )

    # pipeline.save_readouts(fsom,
    #                     '/Users/user/Documents/UNITO_csv_conversion/flowsomtest/',
    #                     'metaclusters',
    #                     threshold_method = 'otsu',
    #                     threshold_report = True
    #                     )

    pipeline.plot_umap_save_readouts_by_fcsfile(
                        fsom,
                        '/Users/user/Documents/UNITO_csv_conversion/flowsomtest/',
                        markers= np.array(config.marker_list),
                        threshold_method = 'otsu',
                        threshold_report = True
                        )

if __name__ == '__main__':
    main()


