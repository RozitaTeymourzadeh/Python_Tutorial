# -*- coding: utf-8 -*-
"""
Created on Aug 12 2021

@author: rozita.teymourzadeh
"""
import math
from typing import List
#from matplotlib import collections

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
15. Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

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
"""
16. 3Sum Closest
Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target.

Return the sum of the three integers.

You may assume that each input would have exactly one solution.

 

Example 1:

Input: nums = [-1,2,1,-4], target = 1
Output: 2
Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
Example 2:

Input: nums = [0,0,0], target = 1
Output: 0
 

Constraints:

3 <= nums.length <= 1000
-1000 <= nums[i] <= 1000
-104 <= target <= 104

"""

def threeSumClosest(self, num: List[int], target: int)-> int:
    best_sum = 100000
    num.sort()

    for i in range(0, len(num)-2):
        if num[i] == num[i-1] and i > 0:
            continue
        lower = i + 1
        upper = len(num) - 1

        while lower < upper:
            sum = num[i] + num[lower] + num[upper]

            if sum == target:
                return sum
            if abs(target - sum ) < abs (target - best_sum):
                best_sum = sum
            if sum <= target:
                lower += 1
                while num[lower] == num[lower-1] and lower < upper:
                    lower += 1
            else:
                upper -= 1

    return best_sum

"""
17. letter combination of the Phone Number

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

Example 1:

Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
Example 2:

Input: digits = ""
Output: []
Example 3:

Input: digits = "2"
Output: ["a","b","c"]
 

Constraints:

0 <= digits.length <= 4
digits[i] is a digit in the range ['2', '9'].

"""

def letterCombinations(self, digits:str) -> List[str]:

    res=[]
    digitToChar = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7":"pqrs",
        "8": "tuv",
        "9": "wxyz"
    }

    def backtrack(i, curStr):
        if len(curStr) == len(digits):
            res.append(curStr)
            return
        for c in digitToChar[digits[i]]:
            backtrack(i+1, curStr + c)
    if digits:
        backtrack(0, "")
    return res

def fourSum(self, nums:List[int], target: int) -> List[List[int]]:
    length = len(nums)
    if length < 4:
        return []
    nums.sort()

    def two_sum(lst, total):
        d = {}
        helper_st = set()
        for i, num in enumerate(lst):
            target_num = total - num
            if target_num not in d:
                d[num] = i
            else:
                helper_st.add((target_num, num))
        return helper_st

    st = set()
    for i in range(length-3):
        for j in range(i+1, length-2):
            n1 = nums[i]
            n2 = nums[j]
            target_total = target - n1 - n2
            helper_res = two_sum(nums[j+1:], target_total)
            if helper_res:
                for x in helper_res:
                    n3, n4 = x
                    st.add((n1, n2, n3, n4))
    return st

'''
Q19. Remove Nth Node From End of List
Given the head of a linked list, remove the nth node from the end of the list and return its head.
Example 1:

Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Example 2:

Input: head = [1], n = 1
Output: []
Example 3:

Input: head = [1,2], n = 1
Output: [1]
 

Constraints:

The number of nodes in the list is sz.
1 <= sz <= 30
0 <= Node.val <= 100
1 <= n <= sz

'''
# class ListNode:
#     def__init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class ListNode(object):
    pass

def removeNthFromEnd(self, head: ListNode, n:int)-> ListNode:
    dummy = ListNode(0, head)
    left = dummy
    right = head

    while n > 0 and right:
        right =  right.next
        n -= 1

    while right:
        left =  left.next
        right = right.next

    left.next = left.next.next
    return dummy.next

'''
Q20 Valid Parentheses

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
 

Example 1:

Input: s = "()"
Output: true
Example 2:

Input: s = "()[]{}"
Output: true
Example 3:

Input: s = "(]"
Output: false
Example 4:

Input: s = "([)]"
Output: false
Example 5:

Input: s = "{[]}"
Output: true
 

Constraints:

1 <= s.length <= 104
s consists of parentheses only '()[]{}'.
'''
def isValid(self, s: str) -> bool:
    stack = []
    closeToOpen = {")":"(","}": "{", "]":"["}

    for c in s:
        if c in closeToOpen:
            if stack and stack[-1] == closeToOpen[c]:
                stack.pop()
            else:
                return False
        else:
            stack.append(c)
    return True if not stack else False

'''
Q21 Merge Two SortedList
Merge two sorted linked lists and return it as a sorted list. 
The list should be made by splicing together the nodes of the first two lists.

 

Example 1:


Input: l1 = [1,2,4], l2 = [1,3,4]
Output: [1,1,2,3,4,4]
Example 2:

Input: l1 = [], l2 = []
Output: []
Example 3:

Input: l1 = [], l2 = [0]
Output: [0]
 

Constraints:

The number of nodes in both lists is in the range [0, 50].
-100 <= Node.val <= 100
Both l1 and l2 are sorted in non-decreasing order.
'''
def mergeTwoLists(self, l1: ListNode , l2: ListNode) -> ListNode:
    dummy = ListNode()
    tail = dummy

    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next

    if l1:
        tail.next = l1
    elif l2:
        tail.next = l2
    return dummy.next

"""
Q22. Generate Parentheses
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

 

Example 1:

Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
Example 2:

Input: n = 1
Output: ["()"]
 

Constraints:

1 <= n <= 8

"""
def generateParenthesis(self, n: int) -> List[str]:

    # only add open paranthesis if open < n
    # only add a closing paranthesis if closed < open
    # valid if open == closed == n

    stack = []
    res = []

    def backtrack(openN, closedN):
        if openN == closedN == n:
            res.append("".join(stack))
            return

        if openN < n:
            stack.append("(")
            backtrack(openN+1, closedN)
            stack.pop()

        if closedN <openN:
            stack.append(")")
            backtrack(openN, closedN +1)
            stack.pop()
    backtrack(0,0)
    return res


'''
Q23: Merge k sorted Lists
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

 

Example 1:

Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
Example 2:

Input: lists = []
Output: []
Example 3:

Input: lists = [[]]
Output: []
 

Constraints:

k == lists.length
0 <= k <= 10^4
0 <= lists[i].length <= 500
-10^4 <= lists[i][j] <= 10^4
lists[i] is sorted in ascending order.
The sum of lists[i].length won't exceed 10^4.
'''
def mergeKLists(self, lists:List[ListNode])-> ListNode:
    if not lists or len(lists) == 0:
        return None

    while len(lists) > 1:
        mergedList = []

        for i in range (0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i+1] if i+1 < len(lists) else None
            mergedList.append(self.mergeList(l1,l2))
        lists = mergedList
    return lists[0]

"""
24. Swap Nodes in Pairs

Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

Example 1:
Input: head = [1,2,3,4]
Output: [2,1,4,3]

Example 2:
Input: head = []
Output: []

Example 3:
Input: head = [1]
Output: [1]
 
Constraints:
The number of nodes in the list is in the range [0, 100].
0 <= Node.val <= 100
"""
def swapPairs(self, head: ListNode) -> ListNode:

    dummy = ListNode(0, head)
    prev, curr = dummy, head

    while curr and curr.next:
        # save node
        nextPair = curr.next.next
        second = curr.next

        # reverse this pair
        curr.next = nextPair
        second.next = curr
        prev.next = second

        #update pointer
        prev = curr
        curr = nextPair
    return dummy.next


"""
Q25. Reverse Nodes in K-Group 

Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.

You may not alter the values in the list's nodes, only nodes themselves may be changed.

Example 1:
Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]

Example 2
Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]

"""
def reverseKGroup(self, head:ListNode, k: int)-> ListNode:
    dummy = ListNode(0, head)
    groupPrev = dummy

    while True:
        kth = self.getKth(groupPrev, k)
        if not kth:
            break
        groupNext = kth.next

        # reverse group
        prev, curr = kth.next, groupPrev.next
        while curr != groupNext:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp # update node
        tmp = groupPrev.next
        groupPrev.next = kth
        groupPrev = tmp
    return dummy.next

    def getKth(self, curr, k):
        while curr and k >0:
            curr = curr.next
            k -= 1
        return curr

""""
Q26 Remove Duplicates from Sorted Array
Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.
Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

Custom Judge:

The judge will test your solution with the following code:
int[] nums = [...]; // Input array
int[] expectedNums = [...]; // The expected answer with correct length

int k = removeDuplicates(nums); // Calls your implementation

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}
If all assertions pass, then your solution will be accepted.
Example 1:

Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
 

Constraints:

0 <= nums.length <= 3 * 104
-100 <= nums[i] <= 100
nums is sorted in non-decreasing order.

"""
def removeDuplicates(self, nums: List[int]) -> int:
    l = 1
    for r in range(1, len(nums)):
        if nums[r] != nums[r-1]:
            nums[l] = nums[r]
            l += 1
    return l

"""
Q27. Remove Element

Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The relative order of the elements may be changed.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

Custom Judge:

The judge will test your solution with the following code:

int[] nums = [...]; // Input array
int val = ...; // Value to remove
int[] expectedNums = [...]; // The expected answer with correct length.
                            // It is sorted with no values equaling val.

int k = removeElement(nums, val); // Calls your implementation

assert k == expectedNums.length;
sort(nums, 0, k); // Sort the first k elements of nums
for (int i = 0; i < actualLength; i++) {
    assert nums[i] == expectedNums[i];
}
If all assertions pass, then your solution will be accepted.

 

Example 1:

Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 2.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums containing 0, 0, 1, 3, and 4.
Note that the five elements can be returned in any order.
It does not matter what you leave beyond the returned k (hence they are underscores).
 

Constraints:

0 <= nums.length <= 100
0 <= nums[i] <= 50
0 <= val <= 100
"""
def removeElement(self, nums: List[int], val:int) -> int:
    k = 0

    for i in range(len(nums)):
        if nums[i] != val:
            nums[k] = nums[i]
            k += 1
    return k

"""
Q28 Implement strStr:
Implement strStr().

Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

Clarification:

What should we return when needle is an empty string? This is a great question to ask during an interview.

For the purpose of this problem, we will return 0 when needle is an empty string. This is consistent to C's strstr() and Java's indexOf().

Example 1:

Input: haystack = "hello", needle = "ll"
Output: 2
Example 2:

Input: haystack = "aaaaa", needle = "bba"
Output: -1
Example 3:

Input: haystack = "", needle = ""
Output: 0
 

Constraints:

0 <= haystack.length, needle.length <= 5 * 104
haystack and needle consist of only lower-case English characters.

"""
def strStr(self, haystack:str, needle: str) ->int:
    if len(needle) == 0:
        return 0
    for i in range(len(haystack) - len(needle) +1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return -1

"""
Q29. Devide Two Integers

Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator.

Return the quotient after dividing dividend by divisor.

The integer division should truncate toward zero, which means losing its fractional part. For example, truncate(8.345) = 8 and truncate(-2.7335) = -2.

Note: Assume we are dealing with an environment that could only store integers within the 32-bit signed integer range: [−231, 231 − 1]. For this problem, assume that your function returns 231 − 1 when the division result overflows.

Example 1:

Input: dividend = 10, divisor = 3
Output: 3
Explanation: 10/3 = truncate(3.33333..) = 3.
Example 2:

Input: dividend = 7, divisor = -3
Output: -2
Explanation: 7/-3 = truncate(-2.33333..) = -2.
Example 3:

Input: dividend = 0, divisor = 1
Output: 0
Example 4:

Input: dividend = 1, divisor = 1
Output: 1
 

Constraints:

-231 <= dividend, divisor <= 231 - 1
divisor != 0

"""
def divide(self, dividend: int , divisor: int) -> int:
    d = abs(dividend)
    dv = abs(divisor)

    output = 0

    while d >= dv:
        tmp = dv
        mul = 1
        while d >= tmp: # This approach is to have log n time complexity, we exponentially subtract tmp from d
            d -= tmp
            output += mul
            mul += mul
            tmp += tmp
        if (dividend < 0 and divisor >=0) or (divisor <0 and dividend >= 0):
            output = - output
        return min(2147483647, max(-2147483648, output))

"""
30. Substring with Concatenation of All Words

You are given a string s and an array of strings words of the same length. Return all starting indices of substring(s) in s that is a concatenation of each word in words exactly once, in any order, and without any intervening characters.

You can return the answer in any order.

Example 1:

Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Explanation: Substrings starting at index 0 and 9 are "barfoo" and "foobar" respectively.
The output order does not matter, returning [9,0] is fine too.
Example 2:

Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
Output: []
Example 3:

Input: s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
Output: [6,9,12]
 

Constraints:

1 <= s.length <= 104
s consists of lower-case English letters.
1 <= words.length <= 5000
1 <= words[i].length <= 30
words[i] consists of lower-case English letters.


"""
"""
31. Next Permutation

Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such an arrangement is not possible, it must rearrange it as the lowest possible order (i.e., sorted in ascending order).

The replacement must be in place and use only constant extra memory.

 

Example 1:

Input: nums = [1,2,3]
Output: [1,3,2]
Example 2:

Input: nums = [3,2,1]
Output: [1,2,3]
Example 3:

Input: nums = [1,1,5]
Output: [1,5,1]
Example 4:

Input: nums = [1]
Output: [1]
 

Constraints:

1 <= nums.length <= 100
0 <= nums[i] <= 100

"""
def swap(self, nums, ind1, ind2):
    temp = nums[ind1]
    nums[ind1] = nums[ind2]
    nums[ind2] =  temp

def reverse(self, nums, beg, end):
    while beg < end:
        self.swap(nums, beg, end)
        beg += 1
        end -= 1

def nextPermutation(self, nums: List[int]) -> None:
    """
    Do not return anything , modify num inplace instead
    """
    if len(nums) == 1:
        return
    if len(nums) == 2:
        return self.swap(nums, 0, 1)
    dec = len(nums) - 2
    # sort ascending
    while dec >= 0 and nums[dec] >= nums[dec + 1]:
        dec -= 1
    self.reverse(nums, dec + 1, len(nums) - 1)
    if dec == -1:
        return
    next_num = dec + 1
    while next_num < len(nums) and nums[next_num] <= nums[dec]:
        next_num += 1
    self.swap(nums, next_num, dec)

"""
32. Longest Valid Parentheses

Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

 

Example 1:

Input: s = "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()".
Example 2:

Input: s = ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()".
Example 3:

Input: s = ""
Output: 0
 

Constraints:

0 <= s.length <= 3 * 104
s[i] is '(', or ')'.
"""

def longestValidParentheses(self, s: str)-> int:
    close, open = 0,0
    maximum = 0
    # Check paranthesise from left to right
    for p in s:
        if p == "(":
            open += 1
        else:
            close += 1
        if close == open:
            maximum = max(maximum, 2 * close)
        elif close > open:
            close, open = 0, 0
    # check paranthesize from right to left
    close, open = 0, 0
    for p in reversed(s):
        if p == ")":
            close += 1
        else:
            open += 1
        if close == open:
            maximum = max(maximum, 2 * close)
        elif open > close:
            close, open = 0,
    return maximum


"""
33. Search in Rotated Sorted Array

There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Example 3:

Input: nums = [1], target = 0
Output: -1
 

Constraints:

1 <= nums.length <= 5000
-104 <= nums[i] <= 104
All values of nums are unique.
nums is an ascending array that is possibly rotated.
-104 <= target <= 104

"""
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # binary search r, l , mid index
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2
            if target == nums[mid]:
                return mid
            # left sorted portion
            if nums[l] <= nums[mid]:
                if target > nums[mid] or target < nums[l]:
                    l = mid + 1
                else:
                    r = mid - 1
            # right sorted portion
            else:
                if target < nums[mid] or target > nums[r]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1

'''
34. Find First and Last Position of Element in Sorted Array

Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.

Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Example 2:

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
Example 3:

Input: nums = [], target = 0
Output: [-1,-1]
 
Constraints:

0 <= nums.length <= 105
-109 <= nums[i] <= 109
nums is a non-decreasing array.
-109 <= target <= 109

'''
def searchRange(self, nums: List[int], target: int) -> List[int]:
    left = self.binSearch(nums, target, True)
    right = self.binSearch(nums, target, False)

    return [left, right]


# leftBias=[True/False], if false, res is rightBiased
def binSearch(self, nums, target, leftBias):
    l, r = 0, len(nums) - 1
    i = -1
    while l <= r:
        m = (l + r) // 2
        if target > nums[m]:
            l = m + 1
        elif target < nums[m]:
            r = m - 1
        else:
            i = m
            if leftBias:
                r = m - 1
            else:
                l = m + 1
    return i

'''
35. Search Insert Position

Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.

Example 1:

Input: nums = [1,3,5,6], target = 5
Output: 2
Example 2:

Input: nums = [1,3,5,6], target = 2
Output: 1
Example 3:

Input: nums = [1,3,5,6], target = 7
Output: 4
Example 4:

Input: nums = [1,3,5,6], target = 0
Output: 0
Example 5:

Input: nums = [1], target = 0
Output: 0
 
Constraints:

1 <= nums.length <= 104
-104 <= nums[i] <= 104
nums contains distinct values sorted in ascending order.
-104 <= target <= 104

'''
def searchInsert(nums: List[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if target == nums[m]:
            return m
        elif target < nums[m]:
            r = m - 1
        else:
            l = m + 1
    return l

"""
Q36) Valid Sudoku

Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
Note:

A Sudoku board (partially filled) could be valid but is not necessarily solvable.
Only the filled cells need to be validated according to the mentioned rules.
 

Example 1:


Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true
Example 2:

Input: board = 
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: false
Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.
 

Constraints:

board.length == 9
board[i].length == 9
board[i][j] is a digit 1-9 or '.'.

"""


def isValidSudoku(self, board: List[List[str]]) -> bool:
    cols = collections.defaultdict(set)
    rows = collections.defaultdict(set)
    squares = collections.defaultdict(set)

    for r in range(9):
        for c in range(9):
            if board[r][c] == ".":
                continue
            if (board[r][c] in rows[r] or
                    board[r][c] in cols[c] or
                    board[r][c] in squares[(r // 3, c // 3)]):
                return False
            cols[c].add(board[r][c])
            rows[r].add(board[r][c])
            squares[(r // 3, c // 3)].add(board[r][c])
    return True

"""
Q37) Sudoku Solver

Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:

Each of the digits 1-9 must occur exactly once in each row.
Each of the digits 1-9 must occur exactly once in each column.
Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
The '.' character indicates empty cells.

 

Example 1:


Input: board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
Output: [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
Explanation: The input board is shown above and the only valid solution is shown below:


 

Constraints:

board.length == 9
board[i].length == 9
board[i][j] is a digit or '.'.
It is guaranteed that the input board has only one solution.

"""
def solveSudoku(self, board: List[List[str]]) -> None:
    def isValid(bard, r, c, num_str):
        for j in range(9):
            if board[r][j] == num_str:
                return False
        for i in range(9):
            if board[i][c] == num_str:
                return False
        x = r // 3
        y = c // 3
        for i in range(3 * x, 3 * x + 3):
            for j in range(3 * y, 3 * y + 3):
                if board[i][j] == num_str:
                    return False
        return True

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == ".":
                for num_str in "123456789":
                    if isValid(board, i, j, num_str):
                        board[i][j] = num_str
                        if self.solveSudoku(board):
                            return True
                        else:
                            board[i][j] = "."
                return False
    return True

"""
Q38. count and say
The count-and-say sequence is a sequence of digit strings defined by the recursive formula:

countAndSay(1) = "1"
countAndSay(n) is the way you would "say" the digit string from countAndSay(n-1), which is then converted into a different digit string.
To determine how you "say" a digit string, split it into the minimal number of groups so that each group is a contiguous section all of the same character. Then for each group, say the number of characters, then say the character. To convert the saying into a digit string, replace the counts with a number and concatenate every saying.

For example, the saying and conversion for digit string "3322251":


Given a positive integer n, return the nth term of the count-and-say sequence.

 

Example 1:

Input: n = 1
Output: "1"
Explanation: This is the base case.
Example 2:

Input: n = 4
Output: "1211"
Explanation:
countAndSay(1) = "1"
countAndSay(2) = say "1" = one 1 = "11"
countAndSay(3) = say "11" = two 1's = "21"
countAndSay(4) = say "21" = one 2 + one 1 = "12" + "11" = "1211"
 

Constraints:

1 <= n <= 30

"""

def countAndSay(n: int) -> str:
    if n == 1:
        return "1"

    prev = countAndSay(n - 1)
    cnt = 1
    res = ""
    for i in range(len(prev)):
        if i == len(prev) - 1 or prev[i] != prev[i+1] :
            res += str(cnt) + prev[i]
            cnt = 1
        else:
            cnt += 1
    return res

if __name__ == "__main__":
    print("Hello")
    # result = isPalindrome(x=121)
    # max_volume = maxArea(height=[1,8,6,2,5,4,8,3,7])
    # result_3sum = threeSumClosest([-1,2,1,-4])
    # print(searchInsert([1, 3, 5, 6], 7))
    print(countAndSay(5))



    # print(solution().letterCombination("34"))
    # print(result)