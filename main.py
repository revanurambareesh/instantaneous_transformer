from globals import *

from utils import *

logger = Logging().get(__name__, args.loglevel)
import torch.backends.cudnn as cudnn

from dataloader import V4V_Dataset
from model import Trainer

from utils import SummaryLogger

def main():
    if os.path.exists(summaries_dir):
        logger.warn(f'Overwriting the exp dir {summaries_dir}')
        import time
        time.sleep(1)
    else:
        os.mkdir(summaries_dir)
        os.mkdir(osj(summaries_dir, 'logs'))

    summary_writer = SummaryLogger(summaries_dir)
    model = Trainer()
    if args.test:
        train_loader = None
    else:
        train_loader = DataLoader(V4V_Dataset(split='training', use_cache=True), batch_size=args.batch_size, num_workers = 18, shuffle=False) #, pin_memory=True)
    
    test_loader = val_loader = DataLoader(V4V_Dataset(split='validation', use_cache=True), batch_size=args.batch_size, num_workers = 8, shuffle=False)

    model.cuda()

    best_err = 99999
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
            args.start_epoch = checkpoint['epoch']
            # best_acc = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    if args.test:
        err = test(test_loader, model, summary_writer, args.start_epoch, 'test')
        summary_dict = {'validation/metrics/err': err}
        summary_writer.log_errors(summary_dict, args.start_epoch)
        exit()

    parameters = model.parameters()
    optimizer = optim.Adadelta(parameters, lr=args.lr) 
    scheduler = None 

    logger.info(f'Args: {args}')

    err = test(val_loader, model, summary_writer, args.start_epoch, 'valid')
    summary_dict = {'validation/metrics/err': err}
    summary_writer.log_errors(summary_dict, 0)

    for epoch in range(args.start_epoch, args.epochs + 1):
        logger.info(f'Epoch {epoch}')
        
        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, optimizer, scheduler, epoch, summary_writer)
        # evaluate on validation set
        if epoch % args.valfreq == 0 and epoch > args.start_epoch:
            err = test(val_loader, model, summary_writer, epoch, 'valid')

            # remember best err and save checkpoint
            is_best = err <= best_err
            best_err = min(err, best_err)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_err,
            }, is_best)

            summary_dict = {'validation/metrics/err': err}
            summary_writer.log_errors(summary_dict, epoch)

    checkpoint = torch.load(osj(weights_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    test_err = test(test_loader, model, summary_writer, checkpoint['epoch'], 'test')



def train(train_loader, model: Trainer, optimizer, scheduler, epoch, summary_writer):
    model.train()

    for idx, (btch, _) in tqdm(enumerate(train_loader)):
        dataX, bpsignal, hrsignal = btch['X'], btch['y_bp'], btch['y_hr']
        dataX, bpsignal, hrsignal = [_.cuda().float() for _ in [dataX, bpsignal, hrsignal]]

        summary_dict = {}

        summary_dict['loss/training/bp_loss'], _ = model.bp_loss(dataX, bpsignal, hrsignal, optimizer, scheduler)
        summary_writer.log_errors(summary_dict, epoch * len(train_loader) + idx)


def test(test_loader, model: Trainer, summary_writer, epoch, phase):
    model.eval()

    with torch.no_grad():
        preds, bvps, gts = [], [], []
        avg_loss = []

        for _, (btch, _) in tqdm(enumerate(test_loader)):
            dataX, bp_y, hr_y = btch['X'], btch['y_bp'], btch['y_hr']
            dataX, bp_y, hr_y = [_.cuda().float() for _ in [dataX, bp_y, hr_y]]
            
            loss, _ = model.bp_loss(dataX, bp_y, hr_y, None, None)
            pred = model.feats['tnet']

            preds.append(det_cpu_npy(pred).astype(np.float32))
            gts.extend(det_cpu_npy(hr_y))
            bvps.extend(det_cpu_npy(bp_y))
            
            avg_loss.append(loss)

        avg = torch.mean(torch.stack(avg_loss))
        summary_dict = {'loss/validation/bp_loss': avg}
        summary_writer.log_errors(summary_dict, epoch)

        return avg


if __name__ == '__main__':
    main()


