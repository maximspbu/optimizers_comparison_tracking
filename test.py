class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        s_p_matrix = [[0] * len(s) for _ in range(len(p))]
        prev_symbol = ""
        was_star_symbol = False
        used_stars_count = 0
        last_one_index = -1
        for p_i in range(len(p)):
            if p[p_i] == "*":
                was_star_symbol = True
                prev_symbol = p[p_i - 1]
            for s_i in range(
                last_one_index + 1, min(p_i + used_stars_count + 1, len(s))
            ):
                if was_star_symbol and (s[s_i] == prev_symbol or prev_symbol == "."):
                    s_p_matrix[p_i][s_i] = 1
                    last_one_index = s_i
                    used_stars_count += 1
                elif (
                    was_star_symbol
                    and p_i >= s_i
                    and p[p_i] != prev_symbol
                    and s_p_matrix[p_i - 1][last_one_index]
                ):
                    was_star_symbol = False
                if (p[p_i] == s[s_i] or p[p_i] == ".") and (
                    p_i == 0 or s_i == 0 or s_p_matrix[p_i - 1][s_i - 1]
                ):
                    s_p_matrix[p_i][s_i] = 1
                    last_one_index = s_i
            prev_symbol = ""
        return bool(s_p_matrix[len(p) - 1][len(s) - 1])

    def wiggleMaxLength(self, nums: list[int]) -> int:
        if len(nums) == 1:
            return 0
        nums_matrix = [[0] * len(nums) for _ in range(len(nums))]
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                nums_matrix[i][j] = nums[j] - nums[i]

        row = 2
        last_sign = 1 if nums[1] - nums[0] > 0 else -1 if nums[1] - nums[0] < 0 else 0
        max_len = 1 if last_sign != 0 else 0
        for i in range(max_len, len(nums) - 1):
            loc_row = row
            while loc_row < len(nums):
                if (
                    nums_matrix[i][loc_row] > 0
                    and (last_sign == -1 or i == 0 and last_sign == 0)
                ) or (
                    nums_matrix[i][loc_row] < 0
                    and (last_sign == 1 or i == 0 and last_sign == 0)
                ):
                    last_sign = (
                        1
                        if nums_matrix[i][loc_row] > 0
                        else -1
                        if nums_matrix[i][loc_row] < 0
                        else 0
                    )
                    max_len += 1
                    loc_row += 1
                    row = loc_row
                    break
                loc_row += 1
        return max_len + 1 if max_len > 0 else 0

    def lengthOfLIS(self, nums: list[int]) -> int:
        def bin_search(arr, value, l, r):
            if l > r:
                return l
            m = (l + r) // 2
            if arr[m] <= value:
                return bin_search(arr, value, m + 1, r)
            return bin_search(arr, value, l, m - 1)

        subs_lengths = [nums[0]]
        for i in range(1, len(nums)):
            if subs_lengths[-1] < nums[i]:
                subs_lengths.append(nums[i])
            else:
                ind = bin_search(subs_lengths, nums[i], 0, len(subs_lengths) - 1)
                subs_lengths[ind] = nums[i]
        return len(subs_lengths)

    def longestValidParentheses(self, s: str) -> int:
        if not s:
            return 0

        n = len(s)
        dp = [0] * n

        max_len = 0

        for i in range(1, n):
            if s[i] == ")":
                if s[i - 1] == "(":
                    dp[i] = 2
                    if i - 2 >= 0:
                        dp[i] += dp[i - 2]

                elif s[i - 1] == ")":
                    prev_open_idx = i - dp[i - 1] - 1

                    if prev_open_idx >= 0 and s[prev_open_idx] == "(":
                        dp[i] = dp[i - 1] + 2
                        if prev_open_idx - 1 >= 0:
                            dp[i] += dp[prev_open_idx - 1]
            max_len = max(max_len, dp[i])

        return max_len

    def threeSumClosest(self, nums: list[int], target: int) -> int:
        nums.sort()
        closest_sum = 0
        min_rest = float("inf")
        for i in range(len(nums) - 2):
            first_num = nums[i]
            new_target = target - first_num
            l = i + 1
            r = len(nums) - 1
            second_num, third_num = nums[l], nums[r]
            while l < r:
                if abs(new_target - (nums[l] + nums[r])) < min_rest:
                    min_rest = abs(new_target - (nums[l] + nums[r]))
                    second_num, third_num = nums[l], nums[r]
                    closest_sum = first_num + second_num + third_num
                if nums[l] + nums[r] < new_target:
                    l += 1
                elif nums[l] + nums[r] > new_target:
                    r -= 1
                else:
                    break
        return closest_sum


s = Solution()
print(
    s.threeSumClosest([10, 20, 30, 40, 50, 60, 70, 80, 90], 1) == 60,
    s.threeSumClosest([10, 20, 30, 40, 50, 60, 70, 80, 90], 1),
)
