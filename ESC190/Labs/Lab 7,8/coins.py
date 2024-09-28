def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)#min(dp[5] =5, dp[5-4]+1 = 2)
            print(dp, i, coin)
    
    return dp[amount] if dp[amount] != float('inf') else -1

if __name__ == "__main__":
    coins = [1, 4, 2]
    amount = 8
    result = coin_change(coins, amount)
    print("Minimum coins required:", result)
