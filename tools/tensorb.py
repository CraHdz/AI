from torch.utils.tensorboard import SummaryWriter

log_dir = "./tools/tblog"
filename_suffix = "log"



def add_scalar(name, Y, X):
    write = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
    write.add_scalar(name, Y, X)

def add_image(img_name, img):
    write = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
    write.add_image(img_name, img)
    write.close

def show_model(model, input_to_model):
    write = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
    write.add_graph(model, input_to_model=input_to_model)
    write.close()