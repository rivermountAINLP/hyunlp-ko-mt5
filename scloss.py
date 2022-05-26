import torch
import torch.nn as nn
import math
import numpy as np

class CEandSCLoss(nn.Module):
  """ Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning(2021) : 
  https://arxiv.org/abs/2011.01403
  아직 초기 코드라 맹신은 금물.
  간단하게 CNN에 돌려보니까 돌아가긴 하는데 loss가 음수로 나오는 등 막장이다보니 추가 수정 예정."""
  
  def __init__(self):
      super(CEandSCLoss, self).__init__()
      self.sum_of_dot_product=0
      self.sum_of_log_softmax=0
      self.sc_subloss=0

  def forward(self, input, target, temperature=0.5, lmbd=0.5):
      # Cross-Entropy Loss Part
      self.celoss=nn.CrossEntropyLoss()(input,target)

      # Supervised Contrastive Loss Part
      self.temperature = temperature
      self.lmbd=lmbd
      if self.temperature<=0: raise ValueError(f'Must input value larger than 0 (got {self.temperature})')
      for i in range(input.size(0)):
        for j in range(input.size(0)):
          # 입력 데이터의 index가 서로 같다면 계산 생략
          if i==j: continue
          # 입력 데이터의 index가 서로 다르지만 레이블이 서로 다르다면 계산 생략
          elif i!=j and target[i]!=target[j]: continue
          # 입력 데이터의 index가 서로 다르고 레이블이 서로 같다면 변형 로그 소프트맥스 계산
          else:
            for k in range(input.size(0)):
              # index가 동일한 경우를 제외하고 나머지 경우만 내적 진행
              if i==k: continue
              else: self.sum_of_dot_product+=torch.dot(input[i],input[k])
              self.sum_of_dot_product*=self.temperature
            dot_product=torch.dot(input[i],input[j])/self.temperature
            # 로그 소프트맥스 계산
            log_softmax=math.log(dot_product/self.sum_of_dot_product)
            # 로그 소프트맥스 값 전부 합산
            self.sum_of_log_softmax+=log_softmax
        # 해당 index의 레이블 개수
        label_count=target.tolist().count(target[i])
        # 레이블이 1개라면 contrastive고 뭐고 할 수 없으므로 이때의 loss는 0
        if label_count==1:self.sce_subloss+=0
        # 레이블이 1개가 아니라면 레이블 개수에 1을 뺀 값으로 나누고 -1을 곱하기
        else: self.sc_subloss+=(-1)*self.sum_of_log_softmax/(label_count-1)
      # 최종 계산 결과를 텐서로 변환
      self.scloss=torch.tensor(self.sc_subloss)
      # CE loss와 SCL을 람다에 대해 가중합
      return (1-self.lmbd)*self.celoss + self.lmbd*self.scloss
