from Agent import Agent
from Helper import getStockData, getState,formatPrice
import pandas as pd
import tqdm
window_size = 100
batch_size = 32
agent = Agent(window_size, batch_size)
data = getStockData("LT.NS")
l = len(data) - 1

episode_count = 200
Buy, Sell, Rewards, Total_Profit = [],[],[],[]
Buy_t, Sell_t, Rewards_t, Total_Profit_t = [],[],[],[]
for e in tqdm.tqdm(range(episode_count)):
    # print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    agent.inventory = []
    total_profit = 0
    done = False
    for t in range(l):
        action = agent.act(state)
        action_prob = agent.actor_local.model.predict(state)

        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        if action == 1:
            agent.inventory.append(data[t])
            # print("Buy:" + formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            # print("sell: " + formatPrice(data[t]) + "| profit: " +
            #       formatPrice(data[t] - bought_price))

        if t == l - 1:
            done = True
        agent.step(action_prob, reward, next_state, done)
        state = next_state

        # if done:
        #     print("------------------------------------------")
        #     print("Total Profit: " + formatPrice(total_profit))
        #     print("------------------------------------------")
        Buy.append(formatPrice(data[t]))
        Sell.append(formatPrice(data[t]))
        Rewards.append(reward)
        Total_Profit.append(formatPrice(total_profit))

test_data = getStockData("LT_test")
l_test = len(test_data) - 1
state = getState(test_data, 0, window_size + 1)


for t in range(l_test):
    action = agent.act(state)
    next_state = getState(test_data, t + 1, window_size + 1)
    reward = 0
    if action == 1:
        agent.inventory.append(test_data[t])
        # print("Buy: " + formatPrice(test_data[t]))
    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        reward = max(test_data[t] - bought_price, 0)
        total_profit += test_data[t] - bought_price
        # print("Sell: " + formatPrice(test_data[t]) + " | profit: " + formatPrice(test_data[t] - bought_price))

    if t == l_test - 1:
        done = True
    agent.step(action_prob, reward, next_state, done)
    state = next_state

    if done:
        print("------------------------------------------")
        print("Total Profit: " + formatPrice(total_profit))
        print("------------------------------------------")
    Buy_t.append(formatPrice(data[t]))
    Sell_t.append(formatPrice(data[t]))
    Rewards_t.append(reward)
    Total_Profit_t.append(formatPrice(total_profit))

df_train = pd.DataFrame(list(zip(Buy_t, Sell_t, Rewards_t,Total_Profit_t)),
              columns=['Buy_t', 'Sell_t', 'Rewards_t','Total_Profit_t'])

df_test = pd.DataFrame(list(zip(Buy, Sell, Rewards,Total_Profit)),
              columns=['Buy', 'Sell', 'Rewards','Total_Profit'])

df_test.to_csv('test_data_output.csv')
df_train.to_csv('train_data_output.csv')



