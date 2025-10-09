var roi = ee.Geometry.Rectangle([146.15, -34.25, 146.55, -33.95]);
Map.centerObject(roi, 11);

var cadastre = ee.FeatureCollection('projects/cs88-468908/assets/polygon')
                  .filterBounds(roi);
Map.addLayer(cadastre, {}, 'Cadastre');

var dateStart = '2024-02-01';
var dateEnd   = '2024-10-31';

var EXPORT_FOLDER = 'GEE_Exports';
var EXPORT_PREFIX = 'S2_RGB8_' + dateStart + '_' + dateEnd;// 

var farmlandThreshold = 0.25; // 25%


var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterDate(dateStart, dateEnd)
  .filterBounds(roi)
  .map(function(img) {
    // SCL 去云（3=Cloud shadow, 8=Cloud medium prob, 9=Cloud high prob, 
    // 10=Thin cirrus）
    var scl = img.select('SCL');
    var cloud = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));
    // Select the required band and perform mask and 
    // scale conversion (DN→ reflectance)
    return img.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
              .updateMask(cloud.not())
              .divide(10000);
  })
  .median()
  .clip(roi);

var EXPORT_CRS = 'EPSG:32755';
var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');

var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate('2024-08-01', '2024-10-31') // Change the timeline if needed
  .filterBounds(cadastre)
  .select('label');

var farmland = dw
  .map(function(img) {
    // 4 = crops, 7 = bare
    return img.eq(4).or(img.eq(7));
  })
  .reduce(ee.Reducer.max())
  .rename('farmland');

Map.addLayer(farmland, {min:0, max:1, palette:['grey','yellow']},
     'DW Farmland');

var farmlandPolygons = cadastre.map(function(poly){
  var farmlandInPoly = farmland.clip(poly.geometry());
  var farmlandPixels = farmlandInPoly.eq(1);
  var stats = farmlandPixels.reduceRegion({
    reducer: ee.Reducer.sum().combine({
      reducer2: ee.Reducer.count(),
      sharedInputs: true
    }),
    geometry: poly.geometry(),
    scale: 10,
    maxPixels: 1e9
  });
  var ratio = ee.Number(stats.get('farmland_sum'))
                .divide(ee.Number(stats.get('farmland_count')));
  return poly.set('farmland_ratio', ratio);
});

var selectedPolygons = farmlandPolygons.filter(ee.Filter.gte('farmland_ratio', 
    farmlandThreshold));
var idList = selectedPolygons.aggregate_array('OBJECTID');

print('The number of plots containing no less than 25% farmland:', 
    selectedPolygons.size());
print('List of ObjectiDs for plots:', idList);
Map.addLayer(selectedPolygons, {color:'red'}, 'Selected Polygons (>=25%)');


// =========================
// Generate 8-bit three-channel RGB (B4/B3/B2) for annotation
// =========================

// Only take the three visible light channels (True Color: B4=R, B3=G, B2=B)
var rgbFloat = s2.select(['B4','B3','B2']); // It's still 0 to 1 floating-point 

// Perform linear stretching using the 2nd to 98th quantiles of AOI, 
// and then convert to 8 bits
var percentiles = rgbFloat.reduceRegion({
  reducer: ee.Reducer.percentile([2, 98]),
  geometry: roi,
  scale: 20,      // 
  bestEffort: true
});

function stretchTo8bit(img, p) {
  var bnames = img.bandNames();
  var stretched = bnames.map(function(b){
    b = ee.String(b);
    var lo = ee.Number(p.get(b.cat('_p2')));
    var hi = ee.Number(p.get(b.cat('_p98')));
    var band = img.select([b]);
    var scaled = band.subtract(lo).divide(hi.subtract(lo))
                     .clamp(0,1).multiply(255).toUint8();
    return scaled.rename(b);
  });
  return ee.ImageCollection.fromImages(stretched).toBands().rename(
    img.bandNames());
}

var rgb8 = stretchTo8bit(rgbFloat, percentiles);


Export.image.toDrive({
  image: rgb8,  // Uint8, 3-band(B4,B3,B2)
  description: EXPORT_PREFIX + '_RGB8',
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_PREFIX + '_RGB8',
  region: roi,
  scale: 10,                 // 10m resolution
  crs: EXPORT_CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
