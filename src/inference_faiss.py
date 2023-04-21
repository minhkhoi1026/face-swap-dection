import os
from lightning_fabric import seed_everything
import numpy as np
from src.dataset.text_pc_dataset import PointCloudOnlyDataset, TextOnlyDataSet
from src.model import MODEL_REGISTRY
from src.utils.opt import Opts
from src.utils.pc_transform import Normalize, ToTensor
import torch
from torch.utils.data import DataLoader
from src.utils.retriever import FaissRetrieval
from torchvision.transforms import transforms
from tqdm import tqdm
import datetime


def inference_loop(model, data_loader, obj_value_key, obj_id_key, device):
    model.eval()
    model.to(device)
    obj_ids = []
    obj_embeddings = []
    print(f"- Extracting embeddings of {obj_value_key}...")
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            obj_emb = model(batch[obj_value_key].to(device))
            obj_embeddings.append(obj_emb.detach().cpu().numpy())
            obj_ids.extend(batch[obj_id_key])

    obj_embeddings = np.concatenate(obj_embeddings, axis=0)
    return obj_embeddings, obj_ids


def save_result_to_csv(filepath, top_k_indexes_all, query_ids, pc_ids):
    folder_name = os.path.dirname(filepath)
    os.makedirs(folder_name, exist_ok=True)
    max_k = top_k_indexes_all.shape[1]

    with open(filepath, "w") as f:
        for i, query_id in enumerate(query_ids):
            f.write(query_id)
            f.write(",")
            for j in range(max_k):
                f.write(pc_ids[top_k_indexes_all[i, j]])
                if j < max_k - 1:
                    f.write(",")
                else:
                    f.write("\n")


def inference(cfg):
    # point cloud dataloader for inference
    test_pc_transforms = transforms.Compose([Normalize(), ToTensor()])
    pc_testset = PointCloudOnlyDataset(
        **cfg["dataset"]["point_cloud"]["params"], pc_transform=test_pc_transforms
    )
    pc_test_loader = DataLoader(
        dataset=pc_testset,
        collate_fn=pc_testset.collate_fn,
        **cfg["data_loader"]["point_cloud"]["params"],
    )

    # text dataloader for inference
    text_testset = TextOnlyDataSet(**cfg["dataset"]["text"]["params"])
    text_test_loader = DataLoader(
        dataset=text_testset,
        collate_fn=text_testset.collate_fn,
        **cfg["data_loader"]["text"]["params"],
    )

    # model for inference
    model = MODEL_REGISTRY.get(cfg["model"]["name"])(cfg)
    model = model.load_from_checkpoint(
        cfg["model"]["pretrained_ckpt"], cfg=cfg, strict=True
    )

    # side-setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retriever = FaissRetrieval(dimension=cfg["model"]["embed_dim"], cpu=True)

    print("- Evaluation started...")

    query_embeddings, query_ids = inference_loop(
        model=model.lang_encoder,
        data_loader=text_test_loader,
        obj_value_key="queries",
        obj_id_key="query_ids",
        device=device,
    )

    pc_embeddings, pc_ids = inference_loop(
        model=model.pc_encoder,
        data_loader=pc_test_loader,
        obj_value_key="point_clouds",
        obj_id_key="point_cloud_ids",
        device=device,
    )

    print("- Calculating similarity...")

    max_k = len(pc_embeddings)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    top_k_scores_all, top_k_indexes_all = retriever.similarity_search(
        query_embeddings=query_embeddings,
        gallery_embeddings=pc_embeddings,
        top_k=max_k,
        query_ids=query_ids,
        gallery_ids=pc_ids,
        save_results=f"temps/query_results_{timestamp}.json",
    )

    print("- Done calculate similarity...")

    print("Saving result as csv file...")
    save_result_to_csv(
        f"temps/query_results_{timestamp}.csv", top_k_indexes_all, query_ids, pc_ids
    )
    print("Saved!")


if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    seed_everything(seed=cfg["global"]["SEED"])
    inference(cfg)
