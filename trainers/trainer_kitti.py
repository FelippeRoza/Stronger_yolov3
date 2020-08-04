from trainers.base_trainer import BaseTrainer
from evaluator.kittieval import EvaluatorKITTI

class Trainer(BaseTrainer):
  def __init__(self, args, model, optimizer,lrscheduler):
    super().__init__(args, model, optimizer,lrscheduler)

  def _get_loggers(self):
    super()._get_loggers()
    self.TESTevaluator = EvaluatorKITTI(anchors=None,
                                      cateNames=self.labels,
                                      rootpath=self.dataset_root,
                                      score_thres=0.01,
                                      iou_thres=self.args.EVAL.iou_thres,
                                      use_07_metric=False,
                                      save_img_dir=self.args.EXPER.save_img_dir,
                                      num_visual=self.args.EXPER.num_visual
                                      )
    self.logger_custom = ['mAP']+['AP@{}'.format(cls) for cls in self.labels]
