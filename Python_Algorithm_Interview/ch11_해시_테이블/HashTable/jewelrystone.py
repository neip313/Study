import collections

class Solution:
    def numJewelsInStones1(self, J: str, S: str) -> int:
        freqs = {}
        count = 0

        # 돌(S)의 빈도 수 계산
        for char in S:
            if char not in freqs:
                freqs[char] = 1
            else:
                freqs[char] += 1

        # 보석(J)의 빈도 수 합산
        for char in J:
            if char in freqs:
                count += freqs[char]

        return count

    def numJewelsInStones2(self, J: str, S: str) -> int:
        freqs = collections.defaultdict(int)
        count = 0

        # 비교 없이 돌(S) 빈도 수 계산
        for char in S:
            freqs[char] += 1

        # 비교 없이 보석(J) 빈도 수 합산
        for char in J:
            count += freqs[char]

        return count

    def numJewelsInStones3(self, J: str, S: str) -> int:
        freqs = collections.Counter(S)  # 돌(S) 빈도 수 계산
        count = 0

        # 비교 없이 보석(J) 빈도 수 합산
        for char in J:
            count += freqs[char]

        return count

    def numJewelsInStones4(self, J: str, S: str) -> int:
        return sum(s in J for s in S)