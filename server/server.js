const express = require('express') // the core framework for handling HTTP requests.
const app = express(); // initialize a new express application
const bodyParser = require('body-parser'); // middleware for parsing incoming request bodies
const port = 4000;
const cors = require('cors'); // middleware to enable cross-origin resource sharing
const fs = require('fs'); // node.js file system module for handling file operations

const jsonfile = require('jsonfile');
let website = ['null'];

app.use(cors({
    credentials: true,
    origin: true
}));
app.use(bodyParser.urlencoded({
    limit: '100mb', extended: true
}));
app.use(bodyParser.json({ limit: '100mb' }));


async function insertApiTraces(newHttpReq, website) {
    const file = 'output/' + website + '/apiTraces.json';
    jsonfile.writeFile(file, newHttpReq, {
        flag: 'a'
    }, function(err) {
        if (err) console.error(err);
    })
}

app.post('/apiTraces', (req, res) => {
    req.body.top_level_url = website[0];
    insertApiTraces(req.body, website[0]);
    res.send("request-success");
})


app.post('/complete', (req, res) => {
    website[0] = req.body.website;
    res.send("complete-success");
})

app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`);
})