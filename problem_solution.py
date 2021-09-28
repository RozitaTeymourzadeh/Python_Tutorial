# -*- coding: utf-8 -*-
"""
Created on Aug 12 2021

@author: rozita.teymourzadeh
"""
import math
from typing import List

"""
3. Longest Substring Without Repeating Characters
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

"""
def lengthOfLongestSubstring(self, s: str) -> int:
    def check(start, end):
        chars = [0] * 128
        for i in range(start, end + 1):
            c = s[i]
            chars[ord(c)] += 1
            if chars[ord(c)] > 1:
                return False
        return True

    n = len(s)

    res = 0
    for i in range(n):
        for j in range(i, n):
            if check(i, j):
                res = max(res, j - i + 1)
    return res

"""
4. Given two sorted arrays nums1 and nums2 of size m and n respectively, 
return the median of the two sorted arrays.
The overall run time complexity should be O(log (m+n)).
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.

"""

def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    x = sorted(nums1 + nums2)
    length = len(x)

    if length == 0:
        median = 0
    elif length == 1:
        median = x[0]
    else:
        median_index = abs(length // 2)
        median = x[median_index]
        if length > 1 and length % 2 == 0:
            median += x[median_index - 1]
            median /= float(2)
    return median

"""
5. Given a string s, return the longest palindromic substring in s.
Input: s = "babad"
Output: "bab"
Note: "aba" is also a valid answer.

"""

def longestPalindrome(self, s: str) -> str:
    self.maxlen = 0
    self.start = 0

    for i in range(len(s)):
        self.expandFromCenter(s, i, i)
        self.expandFromCenter(s, i, i + 1)
    return s[self.start:self.start + self.maxlen]

def expandFromCenter(self, s, l, r):
    while l > -1 and r < len(s) and s[l] == s[r]:
        l -= 1
        r += 1

    if self.maxlen < r - l - 1:
        self.maxlen = r - l - 1
        self.start = l + 1

"""
6. ZigZag Conversion
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: 
(you may want to display this pattern in a fixed font for better legibility)
And then read line by line: "PAHNAPLSIIGYIR"
P   A   H   N
A P L S I I G
Y   I   R

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"

Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I
"""

def convert(self, s: str, numRows: int) -> str:
    if numRows == 1: return s
    res = ""
    for r in range(numRows):
        increment = (numRows - 1) * 2
        for i in range(r,len(s), increment):
            res += s[i]
            if (r > 0 and r < numRows -1 and i + increment - 2 * r < len(s)):
                res += s[i + increment - 2 * r]
    return res



"""
7. Reverse Integer
Given a signed 32-bit integer x, return x with its digits reversed. 
If reversing x causes the value to go outside the signed 32-bit integer 
range [-231, 231 - 1], then return 0.
Input: x = 123
Output: 321
Input: x = 0
Output: 0    
"""

def reverse(self, x: int) -> int:
    MIN = -2**31
    MAX = (2**31) - 1
    res = 0
    while x:
        digit = int(math.fmod(x, 10))
        x = int(x/10)
        if (res > MAX // 10 or (res == MAX //10 and digit >= MAX % 10)):
            return 0
        if (res < MIN //10 or (res == MIN //10 and digit <= MIN % 10)):
            return 0
        res = res * 10 + digit
    return res

"""
8. String to Integer (atoi)
Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer 

Input: s = "42"
Output: 42

Condition:
- whitespace
- +/- symbol
- numbers
- between MAX_INT and MIN_INT constraints
- random characters 
"""

def myAtoi(self, s: str) -> int:

    result = 0
    i = 0
    negative = 1
    MAX_INT = (2 ** 31) - 1
    MIN_INT = -2 ** 31

    # check corner cases
    # Whitespace
    while i < len(s) and s[i] == ' ':
        i += 1

    # Sign
    if i < len(s) and s[i] == '-':
        i += 1
        negative = -1
    elif i <len(s) and s[i] == '+':
        i += 1
        negative = 1
    # number
    checker = set('0123456789')
    while i < len(s) and s[i] in checker:
        # check overflow
        if result > MAX_INT/10 or (result == MAX_INT/10 and int(s[i]) > 7):
            return MAX_INT if negative == 1 else MIN_INT
        result = result * 10 + int(s[i])
        i += 1

    result = result * negative
    # check for max and Min
    if result < 0:
        return max(result, MIN_INT)
    return min(result,MAX_INT)

"""
9. Given an integer x, return true if x is palindrome integer

Input: x = 12
Output: true

Input: x = -121
Output: false

Condition:
- (-) value is not palindrome
- between MAX_INT and MIN_INT constraints -2**31 <= x <= 2**31 - 1

"""
def isPalindrome(x: int) ->bool:

    reverse = 0
    cal = x
    MIN_INT = -2**31
    MAX_INT = 2**31 - 1

    if x > MAX_INT or x < MIN_INT:
        return False

    if x < 0:
        return False

    while cal != 0:
        mod = int(cal % 10)
        cal = int(cal / 10)
        reverse = reverse * 10 + mod

    if x == reverse:
        return True
    return False

"""
10. Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

'.' Matches any single character
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).

Input: s = "aa", p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".

Input: s = "ab", p = ".*"
Output: true
Explanation: ".*" means "zero or more (*) of any character (.)".

Input: s = "aab", p = "c*a*b"
Output: true
Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore, it matches "aab".

Input: s = "mississippi", p = "mis*is*p*."
Output: false

Condition:
1 <= s.length <= 20
1 <= p.length <= 30
s contains only lowercase English letters.
p contains only lowercase English letters, '.', and '*'.
It is guaranteed for each appearance of the character '*', there will be a previous valid character to match.

"""
def isMatch(self, s: str, p: str) -> bool:
    # TOP-Down Memoization
    # Ref: NeetCode
    cache = {}
    def dfs(i,j):
        if (i, j) in cache:
            return cache[(i, j)]
        if i >= len(s) and j >= len(p):
            return True
        if j >= len(p):
            return False

        match = i < len(s) and (s[i] == p[j] or p[j] == ".")
        if (j + 1) < len(p) and p[j + 1] == "*":
            cache[(i, j)] = (dfs(i,j+2) or      #dont use *
                    (match and dfs(i+1, j)))    # use *
            return cache[(i, j)]
        if match:
            cache[(i, j)] = dfs(i+1, j+1)
            return cache[(i, j)]
        cache[(i, j)] = False
        return False

    return dfs(0, 0)

"""
11. Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). 
n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0).
Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

Notice that you may not slant the container.

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. 
In this case, the max area of water (blue section) the container can contain is 49.

Input: height = [1,1]
Output: 1

Input: height = [4,3,2,1,4]
Output: 16

Input: height = [1,2,1]
Output: 2

Condition:
n == height.length
2 <= n <= 105
0 <= height[i] <= 104
"""
def maxArea(self, height: List[int]) -> int:
    # Ref: Google Engineer Explains
    best_volume = 0

    r = 0
    l = len(height) - 1

    while r < l:
        current_volume = min(height[l],height[r]) * (l - r)
        best_volume = max(best_volume, current_volume)
        if height[r] < height[l]:
            r += 1
        else:
            l -= 1
    return best_volume

"""
12. Integer to Roman
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000

For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given an integer, convert it to a roman numeral.

Input: num = 3
Output: "III"

Input: num = 58
Output: "LVIII"
Explanation: L = 50, V = 5, III = 3.

Constraints:

1 <= num <= 3999
"""

def intToRoman(self, num: int) -> str:
    symList = [["I", 1],
               ["IV", 4],
               ["V", 5],
               ["IX", 9],
               ["X", 10],
               ["XL", 40],
               ["L", 50],
               ["XC", 90],
               ["C", 100],
               ["CD", 400],
               ["D", 500],
               ["CM", 900],
               ["M", 1000]]

    res = ""
    for sym, val in reversed(symList):
        if num // val:
            count = int (num //val)
            res += (sym * count)
            num = num % val
    return res

"""
13. Roman to Integer
Easy

1855

141

Add to List

Share
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

Input: s = "III"
Output: 3

Input: s = "IV"
Output: 4

Input: s = "IX"
Output: 9

Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.

Constraints:

1 <= s.length <= 15
s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').
It is guaranteed that s is a valid roman numeral in the range [1, 3999].

"""
def romanToInt(self, s: str)-> int:

    sym = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    res = 0

    for i in range(len(s)):
        if i + 1 < len(s) and sym[s[i]] < sym[s[i+1]]:
            res -= sym[s[i]]
        else:
            res += sym[s[i]]
    return res

"""
14. Longest Common Prefix

Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string "".

Example 1:

Input: strs = ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
 

Constraints:

1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] consists of only lower-case English letters.

"""
def longestCommonPrefix(self, strs: List[str])-> str:
    res=""
    for i in range(len(strs[0])):
        for s in strs:
            if i == len(s) or s[i] != strs[0][i]:
                return res
        res += strs[0][i]
    return res

'''
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Example 2:

Input: nums = []
Output: []
Example 3:

Input: nums = [0]
Output: []
 

Constraints:

0 <= nums.length <= 3000
-105 <= nums[i] <= 105
'''
def threeSum(self, num: List[int])-> List[List[int]]:
    res = []
    num.sort()

    for i, a in enumerate(num):
        if i > 0 and a == num[i - 1]:
            continue
        l, r = i + 1, len(num) - 1
        while l < r:
            sum = a + num[l] + num[r]
            if sum == 0:
                res.append([a, num[l], num[r]])
                l += 1
                while num[l] == num[l - 1] and l < r:
                    l += 1
            elif sum < 0:
                l += 1
            else:
                r -= 1
    return res



if __name__ == "__main__":
    print("Hello")
    result = isPalindrome(x=121)
    max_volume = maxArea(height=[1,8,6,2,5,4,8,3,7])
    print(result)