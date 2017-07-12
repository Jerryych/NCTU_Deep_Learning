from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
import sys


def main():
    mode = sys.argv[1]
    print 'Soft(s) or Hard(h): ', mode

    # load train dataset
    data = load_coco_data(data_path='./data', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='./data', split='val')

    model = CaptionGenerator(mode, word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=10, batch_size=50, update_rule='adam',
                                          learning_rate=0.001, print_every=1000, save_every=1, image_path='./image/',
                                    pretrained_model=None, model_path='model/lstm/', test_model='model/lstm/model-10',
                                     print_bleu=True, log_path='log/')

    solver.train()

    #test_data = load_coco_data(data_path='./data', split='test')
    #solver.test('soft', test_data, split='test')

if __name__ == "__main__":
    main()
