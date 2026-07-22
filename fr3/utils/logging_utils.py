"""Experiment logging integrations."""

import wandb


class WandbLogger:
    """Small adapter around Weights & Biases used by training scripts."""

    def __init__(self, project, exp_name, config):
        wandb.init(project=project, entity="junning", name=exp_name, config=config)

    def log_data(
        self,
        scalar_data_names,
        scalar_datas,
        video_data_config=None,
        img_data_path=None,
        table_data_config=None,
    ):
        data_dict = dict(zip(scalar_data_names, scalar_datas))

        if video_data_config is not None:
            video_data_path, video_type, fps = video_data_config
            data_dict["video"] = wandb.Video(
                str(video_data_path), fps=fps, format=str(video_type)
            )

        if img_data_path is not None:
            data_dict["image"] = wandb.Image(img_data_path)

        if table_data_config is not None:
            table_keys, columns, datas = table_data_config
            for key, column, data in zip(table_keys, columns, datas):
                data_dict[key] = wandb.Table(columns=column, data=data)

        wandb.log(data_dict)
