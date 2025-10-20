function calculateSum(limit) {
    let sum = 0;
    for (let i = 1; i <= limit; i++) {
        sum += i;
    }
    return sum;
}

const limit = 10;
const sum = calculateSum(limit);
console.log(`Sum of numbers from 1 to ${limit} is ${sum}`);
