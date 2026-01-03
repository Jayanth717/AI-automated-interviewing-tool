// Remove only the SVG inside #histo (safer)
d3.select("#histo").selectAll("svg").remove();

// Set the dimensions and margins of the graph
var margin = { top: 20, right: 20, bottom: 30, left: 70 },
    width = 500 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

// Set the ranges
var x = d3.scaleBand()
    .range([0, width])
    .padding(0.1);

var y = d3.scaleLinear()
    .range([height, 0]);

// Append the svg object to the page
var svg = d3.select("#histo")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Load the data (D3 v5+)
d3.csv("/static/js/db/text_perso.txt").then(function(data) {

    // Format the data
    data.forEach(function(d) {
        d.Value = +d.Value || 0;
    });

    // Scale the range of the data
    x.domain(data.map(function(d) { return d.Trait; }));
    y.domain([0, d3.max(data, function(d) { return d.Value; })]);

    // Append the rectangles for the bar chart
    svg.selectAll(".bar")
        .data(data)
        .enter()
        .append("rect")
        .attr("class", "bar")
        .attr("x", function(d) { return x(d.Trait); })
        .attr("width", x.bandwidth())
        .attr("y", function(d) { return y(d.Value); })
        .attr("height", function(d) { return height - y(d.Value); })
        .style("fill", "#b71b1b");

    // Add the X Axis
    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));

    // Add the Y Axis
    svg.append("g")
        .call(d3.axisLeft(y));

}).catch(function(error) {
    console.error("Error loading the data:", error);
});
