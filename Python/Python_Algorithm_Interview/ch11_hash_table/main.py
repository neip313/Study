from HashTable import hashmapdesign
from HashTable import jewelrystone
from HashTable import longest
from HashTable import k_frequency

hash_map = hashmapdesign.MyHashMap()
jewel = jewelrystone.Solution()
longest = longest.Solution()
freq = k_frequency.Solution()

# Question 28
if __name__ == '__main__':
    hash_map.put(1,1)
    hash_map.put(2,2)
    print(hash_map.get(1))
    print(hash_map.get(3))
    hash_map.put(2,1)
    print(hash_map.get(2))
    hash_map.remove(2)
    print(hash_map.get(1))

# Question 29
if __name__ == '__main__':
    print("Q29: ",jewel.numJewelsInStones3("aA", "aAAbbbb"))

# Question 30
if __name__ == '__main__':
    print("1번 문제: ",longest.lengthOfLongestSubstring("abcabcbb"))
    print("2번 문제: ",longest.lengthOfLongestSubstring("bbbbb"))
    print("3번 문제: ",longest.lengthOfLongestSubstring("pwwkew"))

# Question 31
if __name__ == '__main__':
    print("Q31: ",freq.topKFrequent1([1,1,1,2,2,3],2))


