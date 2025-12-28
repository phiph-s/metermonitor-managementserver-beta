<template>
  <div class="bg">
    <apexchart
      v-if="series && series.length > 0"
      width="100%"
      height="300"
      type="area"
      :options="chartOptions"
      :series="series"
    ></apexchart>
    <div v-else style="text-align: center; padding: 20px;">
      No history data available
    </div>
  </div>
  <div class="bg" v-if="confidenceSeries && confidenceSeries.length > 0">
    <apexchart
      width="100%"
      height="200"
      type="line"
      :options="confidenceChartOptions"
      :series="confidenceSeries"
    ></apexchart>
  </div>
</template>

<script setup>
import { computed, defineProps } from 'vue';

const props = defineProps({
  history: {
    type: Object,
    default: null
  }
});

const sortedHistory = computed(() => {
  if (!props.history || !props.history.history) return [];
  return [...props.history.history].sort((a, b) => new Date(a[1]) - new Date(b[1]));
});

const series = computed(() => {
  if (sortedHistory.value.length === 0) return [];

  const data = sortedHistory.value.map(item => {
    // item is [value, timestamp, confidence, manual]
    return {
      x: new Date(item[1]).getTime(),
      y: item[0]
    };
  });

  return [{
    name: 'Value',
    data: data
  }];
});

const confidenceSeries = computed(() => {
  if (sortedHistory.value.length === 0) return [];

  const data = sortedHistory.value.map(item => {
    // item is [value, timestamp, confidence, manual]
    return {
      x: new Date(item[1]).getTime(),
      y: item[2]
    };
  });

  return [{
    name: 'Confidence',
    data: data
  }];
});

const chartOptions = {
  chart: {
    id: 'meter-history',
    type: 'area',
    zoom: {
      enabled: true,
      autoScaleYaxis: true
    },
    toolbar: {
      autoSelected: 'zoom'
    }
  },
  xaxis: {
    type: 'datetime'
  },
  stroke: {
    curve: 'stepline',
    width: 2
  },
  dataLabels: {
    enabled: false
  },
  tooltip: {
    x: {
      format: 'dd MMM yyyy HH:mm'
    }
  },
  fill: {
    type: 'gradient',
    gradient: {
      shadeIntensity: 1,
      opacityFrom: 0.7,
      opacityTo: 0.9,
      stops: [0, 100]
    }
  },
  theme: {
    mode: 'dark',
    palette: 'palette1'
  }
};

const confidenceChartOptions = {
  chart: {
    id: 'meter-confidence',
    type: 'line',
    zoom: {
      enabled: true,
      autoScaleYaxis: true
    },
    toolbar: {
      autoSelected: 'zoom'
    }
  },
  xaxis: {
    type: 'datetime'
  },
  stroke: {
    curve: 'smooth',
    width: 2
  },
  dataLabels: {
    enabled: false
  },
  tooltip: {
    x: {
      format: 'dd MMM yyyy HH:mm'
    }
  },
  theme: {
    mode: 'dark',
    palette: 'palette2'
  },
  yaxis: {
    min: 0,
    max: 1
  }
};
</script>

<style scoped>
.bg {
  background-color: rgba(240, 240, 240, 0.1);
  padding: 20px;
  border-radius: 15px;
  margin-bottom: 15px;
}
</style>
