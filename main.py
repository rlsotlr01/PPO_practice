import gym
import torch
from torch.distributions import Categorical
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
T_horizon = 20


# PPO Class
# torch 안의 neural network module을 상속한다.
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        # 데이터를 담을 버퍼를 만든다.
        self.data = []
        # 할인율 감마를 지정한다
        self.gamma = 0.98
        # lambda: GAE 에 쓰이는 변수
        self.lmbda = 0.95
        # eps: Clipping을 위한 변수
        self.eps = 0.1
        # 지정 Time step의 데이터를 몇번 학습을 할 것인지
        self.K = 3

        # 그리고 PPO에 필요한 신경망들을 구축한다.
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)

    # PPO - Policy, Value Network 코드
    # 정책의 출력값을 계산하는 함수
    # input : 상태 x, 배치처리/시뮬레이션 여부 softmax_dim
    # softmax_dim = 0 -> 시뮬레이션 돌릴 때
    # softmax_dim = 1 -> 훈련 돌릴 때, (배치 병렬처리)
    def pi(self, x, softmax_dim=0):
        # 첫번째 신경망 값에 relu 활성화 함수 태운다.
        x = F.relu(self.fc1(x))
        # 그리고 정책 층에 해당 값을 넣어 정책값 출력
        x = self.fc_pi(x)
        # 정책에 대한 확률값을 softmax 태워 활성화시킴.
        prob = F.softmax(x, dim=softmax_dim)
        # 정책 출력
        return prob

    # 상태가치함수의 출력값을 계산하는 함수
    # 인풋 : 상태 x
    def v(self, x):
        # 첫번째 층에 상태를 넣고 relu 활성화 함수 태운다.
        x = F.relu(self.fc1(x))
        # 그리고 상태가치 신경망에 태워 상태가치값 v에 대한 회귀값을 얻는다.
        v = self.fc_v(x)
        # 상태가치를 출력한다.
        return v

    # PPO - Data Processing 부분 코드
    # 데이터 넣기
    def put_data(self, item):
        self.data.append(item)

    # 데이터 전처리해서 배치 만들기
    def make_batch(self):
        # 각각의 종류의 데이터를 각각의 배치에 담기 위해 빈 리스트 준비
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        # 데이터 배치 안의 각각의 데이터를 불러온다.
        for item in self.data:
            s, a, r, s_prime, prob_a, done = item
            # 상태는 [1,2,3,4,6] 이렇게 리스트 형태로 되어 있음
            s_lst.append(s)
            # a, r은 숫자형식, 1 or 2 처럼 되어 있어서 [1] or [2]로 변환
            # 안바꾸면 shape error
            a_lst.append([a])
            r_lst.append([r])
            # 또는 나중에 torch.unsqueeze(텐서리스트, dim=1) 로 바꿀 수도 있음.
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        # 모든 배치 데이터들을 텐서 형식으로 변환해준다.
        s, a, r, s_prime, done_mask, prob_a =   torch.tensor(s_lst, dtype=torch.float),\
                                                torch.tensor(a_lst), \
                                                torch.tensor(r_lst), \
                                                torch.tensor(s_prime_lst, dtype=torch.float),\
                                                torch.tensor(done_lst, dtype=torch.float),\
                                                torch.tensor(prob_a_lst)
        # 이제 데이터 배치는 필요 없으니 비워주고
        self.data = []
        # 배치들을 출력한다.
        return s, a, r, s_prime, done_mask, prob_a

    def train(self):
        # 데이터 배치 전처리를 한다
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        # delta 연산
        # 같은 데이터 배치로 K 에포크수 만큼 학습을 진행한다.
        for i in range(self.K):
            # 다음 상태 배치로 타겟값을 구한다.
            td_target = r + self.gamma*self.v(s_prime)*done_mask
            # 델타를 계산해낸다. (r+V(s_prime)-V(s))
            delta = td_target - self.v(s)
            # 그리고 delta 는 목표값이므로 이를 손실함수에 반영하지 않기 위해
            # detach 를 한다.
            # (만약 detach 안하면 타겟값도 계속 바뀜.)
            delta = delta.detach().numpy()

            # delta 값을 통한 GAE 연산
            advantage_lst = []
            # GAE Advantage 함수를 담을 리스트 지정
            advantage = 0.0
            # 초기값 0 설정
            for delta_t in delta[::-1]:
                # delta_(T-1)부터 계산 시작.
                advantage = self.gamma*self.lmbda*advantage + delta_t[0]
                # delta 값 계산 후 하나하나 리스트에 넣어준다.
                advantage_lst.append([advantage])
            # 마지막으로 해당 리스트를 다시 거꾸로 돌린다.
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # PPO - Clipped Loss 부분
            # 정책값 계산
            pi = self.pi(s, softmax_dim=1)
            # 해당 행동들에 대한 확률값 추출
            pi_a = pi.gather(1, a)
            # 새로운 정책과 오래된 정책 사이의 IS ratio 계산
            ratio = torch.exp(torch.log(pi_a)-torch.log(prob_a))

            # IS ratio를 고려한 advantage loss 계산(정책 손실함수)
            surr1 = ratio*advantage
            # 정책 손실함수를 clipping.
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)*advantage
            # 최종 비용함수 = 정책 손실함수 + 가치 손실함수.
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(td_target.detach(),self.v(s))
            # 여기서 가치 손실함수는 타겟 가치값 사이의 거리
            # (가치 업데이트는 타겟가치값으로 가까워지 것이 목표이므로)

            # * 여기서 짚고 넘어가야 할 클리핑의 의미
            #   만약 IS ratio 가 2.0이고, 클리핑이 1.1이면 해당 2.0에 대한 경사는 거의 무시가 됨.
            #   그 말은 즉, 알고리즘이 형편없는 샘플에 대해서,
            #   즉 강화학습에 도움이 되지 않는 샘플들을 클리핑을 통해 알아서 걸러낸다는 의미이다.

            # * advantage 값은 상수인데 .detach()가 안붙어있다?
            #   잘 보면 delta 연산을 할 때 delta 를 numpy로 변환해주고,
            #   다 계산이 끝난 후에 텐서화 시켜줌.
            #   그러므로 advantage 값은 신경망과 연결되어 있지 않은 상관없는 값이므로,
            #   어차피 상수로 취급이 된다.

            # 최적화기로 최적화를 하기에 앞서 최적화기를 초기화해준다.
            self.optimizer.zero_grad()
            # 손실함수를 통해 경사를 계산한다. (backward : 해당 파라미터들의 경사계산)
            loss.mean().backward()
            # 신경망 매개변수들을 갱신한다.
            self.optimizer.step()

def main():
    # 강화학습을 돌릴 환경을 gym으로 만든다.
    env = gym.make('CartPole-v1')
    # 강화학습 모델을 생성한다.
    model = PPO()
    # 할인율 지정
    gamma = 0.99
    # PPO의 업데이트 간격을 설정
    T = 20
    # score 은 점수를 기록하기 위한 변수
    score = 0.0
    print_interval = 20
    scores = []

    # Main 2 code
    # 10000번의 에피소드를 돌린다.
    for n_epi in range(1000):
        # 환경을 리셋하여 초기 상태를 얻는다.
        s = env.reset()
        # 게임 종료 여부를 False로 초기화
        done = False
        # 게임이 안끝나는 동안 루프를 계속 돌린다.
        while not done:
            # PPO를 학습시킬 T만큼 게임을 돌린다.
            for t in range(T):
                # PPO.pi - 정책 : 상태 s를 넣으면 행동확률분포를 돌려준다.
                prob = model.pi(torch.from_numpy(s).float())
                # Categorical : 확률 값을 Categorical 데이터로 변환해준다.
                # e.g. [0.6, 0.3, 0.1] -> Categorical([0.6, 0.3, 0.1])
                m = Categorical(prob)
                # Categorical([0.6, 0.3, 0.1]) -> 0 (확률 가장 높은 행동 출력)
                a = m.sample().item()
                # 행동을 환경에 넣어 다음 상태와 보상, 게임종료여부와 기타정보를 받아낸다.
                s_prime, r, done, info = env.step(a)
                # 데이터를 집어 넣는다.
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                # 다음 상태를 기존 상태변수에 넣는다.
                s = s_prime
                # 에이전트의 점수칸에 보상을 더해준다.
                score += r
                # 게임이 끝날 시
                if done:
                    # 루프를 나가도록 한다.
                    break
            # 강화학습 모델을 학습시킨다.
            model.train()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            scores.append(score)
            score = 0.0
    plt.plot(scores)
    plt.show()
    plt.savefig('result_main1.png', dpi=300)
    env.close()

if __name__ == '__main__':
    main()