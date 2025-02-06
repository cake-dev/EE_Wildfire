// This file should be used in the GEE interface. It creates a task to 
// download the CSV file containing all fires with a given minimum size
// in a given year from the GlobFire database.

// Rough outlines of the contiguous USA
var geometry =ee.Geometry.Polygon(
  [[[-125.1803892906456, 35.26328285844432],
  [-117.08916345892665, 33.2311514593429],
  [-114.35640058749676, 32.92199940444295],
  [-110.88773544819885, 31.612036247094473],
  [-108.91086200144109, 31.7082477979397],
  [-106.80030780089378, 32.42079476218232],
  [-103.63413436750255, 29.786401496314422],
  [-101.87558377066483, 30.622527701868453],
  [-99.40039768482492, 28.04018292597704],
  [-98.69085295525215, 26.724810345780593],
  [-96.42355704777482, 26.216515704595633],
  [-80.68508661702214, 24.546812350183075],
  [-75.56173032587596, 26.814533788629998],
  [-67.1540159827795, 44.40095539443753],
  [-68.07548734644243, 46.981170472447374],
  [-69.17500995805074, 46.98158998130476],
  [-70.7598785138901, 44.87172183866657],
  [-74.84994741250935, 44.748084983808],
  [-77.62168256782745, 43.005725611950055],
  [-82.45987924104175, 41.41068867019324],
  [-83.38318501671864, 42.09979904377044],
  [-82.5905167831457, 45.06163491639556],
  [-84.83301910769038, 46.83552648258547],
  [-88.26350848510909, 48.143646480291835],
  [-90.06706251069104, 47.553445811024204],
  [-95.03745451438925, 48.9881557770297],
  [-98.45773319567587, 48.94699366043251],
  [-101.7018751401119, 48.98284560308372],
  [-108.43164852530356, 48.81973606668503],
  [-115.07339190755627, 48.93699058308441],
  [-121.82530604190744, 48.9830983403776],
  [-122.22085227110232, 48.63535795404536],
  [-124.59504332589562, 47.695726563030405],
  [-125.1803892906456, 35.26328285844432]]]);

var fc = ee.FeatureCollection(geometry);

// Map functions
var computeArea = function (f) {
  return f.set({'area': f.area()});
}
var computeCentroid = function (f) {
  return f.set({'lon': f.geometry().centroid().coordinates().get(0), 'lat': f.geometry().centroid().coordinates().get(1)});
}
var computeDate = function (f) {
  return f.set({'start_date': ee.Date(f.get('IDate')), 'end_date': ee.Date(f.get('FDate'))});
}

// Start generate all fires. Change the year here, to generate fires for a different year.
var year = '2018';
var min_size = 1e7;

// Find fires in the GlobFire database that are in the contiguous USA and in the given year.
var polygons = ee.FeatureCollection('JRC/GWIS/GlobFire/v2/FinalPerimeters')
                  .filter(ee.Filter.gt('IDate', ee.Date(year+'-01-01').millis()))
                  .filter(ee.Filter.lt('IDate', ee.Date(year+'-12-31').millis()))
                  .filterBounds(geometry);
                  

// Filter out all the Invalid large areas (Infinity), and small wildfires.
polygons = polygons.map(computeArea);
polygons = polygons.filter(ee.Filter.gt('area', min_size)).filter(ee.Filter.lt('area', 1e20));
polygons = polygons.map(computeCentroid).map(computeDate);

// Generate task to download the CSV file. Needs to be clicked on the task tab.
Export.table.toDrive({
    collection: polygons, 
    description: 'us_fire_'+year+'_'+ String(min_size),
    fileFormat: 'csv'
  })
